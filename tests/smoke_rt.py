"""Opt-in tiny-scene RT integration check; no challenge accuracy claim."""

import argparse
import gc
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_core import CONFIGS, GainAccumulator, batch_plan, switches
from rt_backend import RTBackend
from terrain_mesh import GridProjector, triangulate


def main(backend_name):
    metadata = {
        "resolution_m_per_px": 1., "bands_hz": {"58": 5775000000.},
        "simulation": {"tx_height_m_agl": 3., "rx_height_m_agl": 1., "tx_power_dbm": 20.,
            "bandwidth_hz": {"58": 80000000.},
            "material": {"thickness_m": 10., "relative_permittivity": 3.411,
                "conductivity_s_per_m": {"58": .009324}, "scattering_coefficient": .632,
                "xpd_coefficient": .1, "scattering_pattern": "lambertian"}},
    }
    height = np.zeros((8, 8), np.float64)
    vertices, faces, centers, areas = triangulate(height, 1.)
    projector = GridProjector(centers, height.shape, 1.)
    backend = RTBackend(backend_name)
    scene, measurement = backend.scene(vertices, faces, height, (3, 3), metadata, "58")
    print(backend.versions, flush=True)
    # Validate global launch directions, including a non-divisible last batch.
    from sionna.rt.utils import fibonacci_lattice
    total = 20003
    reference = np.asarray(fibonacci_lattice(total).numpy())
    pieces = [np.asarray(backend.lattice_slice(total, b["offset"], b["samples"]).numpy())
              for b in batch_plan(total, 6000, 7)]
    np.testing.assert_allclose(np.concatenate(pieces, axis=1), reference, rtol=0, atol=2e-7)
    # LOS and pure specular transport are deterministic for these switches.
    for flags in (switches("los_only"), {"los": True, "specular_reflection": True,
                                      "diffuse_reflection": False, "refraction": False}):
        reference_gain = backend.solve(scene, measurement, total, 7, 3, flags)
        accumulated = GainAccumulator((len(faces),))
        for batch in batch_plan(total, 6000, 7):
            partial = backend.solve(scene, measurement, batch["samples"], batch["seed"], 3,
                                    flags, total_samples=total, sample_offset=batch["offset"])
            accumulated.add(partial, batch["samples"])
        np.testing.assert_allclose(accumulated.mean(), reference_gain, rtol=3e-5, atol=1e-12)
    print("Global-lattice slicing and weighted LOS/specular batch checks passed.", flush=True)
    for seed in (0, 1):
        for case in CONFIGS:
            gain = backend.solve(scene, measurement, 20000, seed, 3, switches(case))
            if gain.shape != (len(faces),) or not np.isfinite(gain).all() or (gain < 0).any():
                raise AssertionError(f"Invalid gain for {case}")
            if case in ("full", "los_only") and not (gain > 0).any():
                raise AssertionError(f"No direct coverage in a flat test scene for {case}")
            grid = projector.apply(gain)
            if not np.isfinite(grid[projector.footprint]).all():
                raise AssertionError("Invalid grid interpolation")
            print(f"seed={seed} {case}: positive cells={np.count_nonzero(gain)}/{len(gain)}, "
                  f"area-weighted mean gain={np.average(gain, weights=areas):.6g}", flush=True)
    print("RT integration check passed; this is not a mechanism-contribution estimate.")
    backend.release_scene(scene, measurement)
    del scene, measurement
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "llvm"), default="cuda")
    main(parser.parse_args().backend)
