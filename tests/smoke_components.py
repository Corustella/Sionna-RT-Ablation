"""Real RT regression: original full gain, all history groups, paths and cap guard."""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablation_core import batch_plan, switches
from component_backend import ComponentBackend
from component_core import ComponentAccumulator, validate_components
from terrain_mesh import triangulate


def main(args):
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    backend = ComponentBackend(args.backend)
    rows, cols = np.indices((12, 12))
    height = 4 * np.exp(-((rows-6)**2 + (cols-6)**2)/5)
    vertices, faces, _, _ = triangulate(height, 1.)
    # Thin low-loss material deliberately exercises transmission families;
    # dataset generation always keeps the original metadata material unchanged.
    test_metadata = copy.deepcopy(metadata)
    test_metadata["simulation"]["material"]["thickness_m"] = .01
    test_metadata["simulation"]["material"]["conductivity_s_per_m"]["58"] = .00001
    scene, measurement = backend.scene(vertices, faces, height, [2, 2], test_metadata, "58")
    try:
        accumulator = ComponentAccumulator(len(faces))
        reference_sum = np.zeros(len(faces))
        for batch in batch_plan(40000, 13000, 0):
            gain, hits, positive = backend.solve_components(scene, measurement, batch["samples"], batch["seed"], 6,
                total_samples=40000, sample_offset=batch["offset"])
            reference = backend.solve(scene, measurement, batch["samples"], batch["seed"], 6, switches("full"),
                total_samples=40000, sample_offset=batch["offset"])
            np.testing.assert_allclose(gain[0], reference, rtol=1e-5, atol=0)
            accumulator.add(gain, hits, positive, batch["samples"])
            reference_sum += reference * batch["samples"]
        gain = accumulator.mean()
        np.testing.assert_allclose(gain[0], reference_sum/40000, rtol=1e-5, atol=0)
        assert np.all(gain[1:].sum(axis=1) > 0), "Fixture did not exercise every family"
        print("all-family closure:", validate_components(gain, accumulator.hits, accumulator.positive_hits), flush=True)
        print("family hit sums:", accumulator.hits.sum(axis=1).tolist(), flush=True)
        los, _, _ = backend.solve_components(scene, measurement, 20000, 3, 0, flags=switches("los_only"))
        assert not np.any(los[2:]), "LOS-only acquired non-LOS history"
        geometry = backend.geometry(scene, measurement)
        assert geometry["terrain_vertices"].dtype == np.float32
        visible = backend.centroid_los(scene, geometry["cell_centers"])
        assert visible.any() and (~visible).any(), "Fixture must include LOS and shadow"
        positions = geometry["cell_centers"][[0, 40, 100, 200]].astype(np.float32)
        paths = backend.point_paths(scene, positions, 0, 2000, 6, 20000)
        assert np.any(paths["valid"])
        assert paths["vertices"].shape[-1] == 3
        assert paths["interactions"].shape == paths["primitives"].shape
        assert not scene.receivers, "Receiver cleanup failed"
        try:
            backend.point_paths(scene, positions, 0, 2000, 6, 4)
        except RuntimeError as error:
            assert "cap reached" in str(error)
        else:
            raise AssertionError("Saturated candidate buffer was not rejected")
        assert not scene.receivers, "Receiver cleanup after failure failed"
        print("point-path shapes:", {k: list(paths[k].shape) for k in ("a_real", "valid", "interactions", "vertices")}, flush=True)
        # Evaluate bounded chunks at offsets beyond signed 32-bit, without a 4B allocation.
        for offset in (0, 3999999000):
            chunk, hits, positive = backend.solve_components(scene, measurement, 1000, 8, 6,
                total_samples=4000000000, sample_offset=offset)
            validate_components(chunk, hits, positive)
        print("Original full gain regression, all five families, LOS-only, point paths, cap rejection, 4B offsets: PASS")
    finally:
        backend.release_scene(scene, measurement)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--backend", choices=("cuda", "llvm"), default="cuda")
    main(p.parse_args())
