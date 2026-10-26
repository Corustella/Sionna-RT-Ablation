"""Run a paired, fixed-budget Sionna RT mechanism ablation (5.775 GHz by default)."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from ablation_core import CONFIGS, GainAccumulator, batch_plan, switches, write_json, write_npz
from lunar_data import LunarData
from terrain_mesh import GridProjector, triangulate


def code_version():
    root = Path(__file__).resolve().parent
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def hardware_info():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        gpu = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        gpu = None
    return {"nvidia_smi": gpu, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def make_plan(args, data):
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Use at least two different RT --seeds for paired analysis")
    if len(set(args.bands)) != len(args.bands):
        raise ValueError("Duplicate bands")
    if args.rays_per_batch > 2**31 - 1:
        raise ValueError("Use smaller ray batches; samples-per-tx may exceed this limit")
    sim = data.metadata["simulation"]
    if "1.1.0" not in sim["engine"] or sim["propagation_mechanisms"].get("diffraction", False):
        raise ValueError("This runner is for metadata declaring Sionna RT 1.1.0 without diffraction")
    depth = int(sim["max_depth"] if args.max_depth is None else args.max_depth)
    if depth < 1:
        raise ValueError("A mechanism ablation needs max-depth >= 1")
    indices = data.select(args.sample_ids, args.terrains, args.tx_per_terrain, args.selection_seed)
    samples = []
    for index in indices:
        data.sample(index, args.bands)  # Validate the small selection before importing RT.
        samples.append({"index": index, **data.rows[index]})
    batches = {str(seed): batch_plan(args.samples_per_tx, args.rays_per_batch, seed) for seed in args.seeds}
    return {
        "schema": 2, "scope": "metadata_based_reconstruction_not_original_generator",
        "data_root": str(data.root), "split": args.split, "data_files": data.provenance(args.bands),
        "metadata": data.metadata, "samples": samples, "bands": args.bands,
        "seeds": args.seeds, "samples_per_tx": args.samples_per_tx,
        "rays_per_batch": args.rays_per_batch, "batches": batches,
        "launch_sampling": "disjoint_index_ranges_of_one_global_upstream_fibonacci_lattice",
        "seed_role": "interaction_sampling_only_not_initial_launch_directions",
        "max_depth": depth, "rr_depth": -1, "stop_threshold": None,
        "backend": args.backend, "configurations": {key: switches(key) for key in CONFIGS},
        "mesh": {"diagonal": args.diagonal, "coordinates": "x=column*spacing,y=row*spacing,z=height",
                 "vertex_normals": False, "extent": "(W-1)*spacing by (H-1)*spacing",
                 "boundary": "open_heightfield_no_sidewalls_or_bottom",
                 "rx_surface": "same_mesh_translated_in_z_by_metadata_rx_height"},
        "aggregation": "ray_count_weighted_linear_gain_mean_including_zero_estimates",
        "grid_projection": "fixed_XY_centroid_Delaunay_linear_gain_no_fill_no_flip",
        "count": {"samples": len(samples), "terrains": len({s['terrain_id'] for s in samples}),
                  "case_runs": len(samples) * len(args.bands) * len(args.seeds) * len(CONFIGS),
                  "total_launched_rays": len(samples) * len(args.bands) * len(args.seeds) * len(CONFIGS) * args.samples_per_tx},
    }


def run_case(backend, scene, measurement, projector, cells, plan, band, sample, seed, config, out):
    case_dir = out / "maps" / sample["sample_id"] / band / f"seed{seed}"
    final = case_dir / f"{config}.npz"
    schedule = plan["batches"][str(seed)]
    if final.exists():
        with np.load(final, allow_pickle=False) as result:
            if int(result["samples"]) != plan["samples_per_tx"] or result["cell_gain"].shape != (cells,):
                raise ValueError(f"Existing result does not match this plan: {final}")
        print(f"skip {sample['sample_id']} {band} seed={seed} {config}", flush=True)
        return
    checkpoint = case_dir / f"{config}.progress.npz"
    accumulator = GainAccumulator((cells,))
    completed, elapsed = 0, 0.0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            completed = int(saved["completed_batches"])
            accumulator.samples = int(saved["samples"])
            accumulator.weighted = np.asarray(saved["weighted_gain"], np.float64)
            elapsed = float(saved["elapsed_seconds"])
        if (not 0 <= completed <= len(schedule)
                or accumulator.samples != sum(b["samples"] for b in schedule[:completed])
                or accumulator.weighted.shape != (cells,)
                or not np.isfinite(accumulator.weighted).all()
                or (accumulator.weighted < 0).any()):
            raise ValueError(f"Invalid checkpoint: {checkpoint}")
    for i in range(completed, len(schedule)):
        batch = schedule[i]
        started = time.perf_counter()
        gain = backend.solve(scene, measurement, batch["samples"], batch["seed"],
                             plan["max_depth"], switches(config), total_samples=plan["samples_per_tx"],
                             sample_offset=batch["offset"])
        accumulator.add(gain, batch["samples"])
        elapsed += time.perf_counter() - started
        write_npz(checkpoint, weighted_gain=accumulator.weighted, samples=accumulator.samples,
                  completed_batches=i + 1, elapsed_seconds=elapsed)
        print(f"{sample['sample_id']} {band} seed={seed} {config}: "
              f"batch {i+1}/{len(schedule)}, rays={accumulator.samples}, compute={elapsed:.2f}s", flush=True)
    gain = accumulator.mean()
    write_npz(final, cell_gain=gain, grid_gain=projector.apply(gain),
              samples=accumulator.samples, elapsed_seconds=elapsed)


def main(args):
    data = LunarData(args.data_root, args.split)
    plan = make_plan(args, data)
    print(json.dumps({"scope": plan["scope"], "count": plan["count"], "bands": plan["bands"],
                      "seeds": plan["seeds"], "max_depth": plan["max_depth"],
                      "sample_ids": [s["sample_id"] for s in plan["samples"]],
                      "metadata_rays_per_tx": data.metadata["simulation"]["samples_per_tx"],
                      "requested_rays_per_tx": args.samples_per_tx}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("Dry run: inputs validated; no RT import, simulation, or output directory created.")
        return
    from rt_backend import RTBackend
    backend = RTBackend(args.backend)
    out = Path(args.out).resolve()
    version = code_version()
    if args.resume and (version["dirty"] or version["commit"] is None):
        raise ValueError("Resume requires a committed, clean source checkout; uncommitted changes cannot be verified")
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists: {out}; use the identical command with --resume")
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if previous["experiment"] != plan or previous["code"] != version or previous["runtime"] != backend.versions:
            raise ValueError("Resume requires identical code, data identity, runtime and experiment settings")
    else:
        out.mkdir(parents=True)
        write_json(out / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                   "experiment": plan, "code": version, "runtime": backend.versions,
                   "python": sys.version, "platform": platform.platform(), "hardware": hardware_info(), "command": sys.argv})
    started = time.perf_counter()
    projector = None
    for sample in plan["samples"]:
        item = data.sample(sample["index"], args.bands)
        height = item["height_m"]
        vertices, faces, centers, areas = triangulate(height, data.metadata["resolution_m_per_px"], args.diagonal)
        if projector is None:
            projector = GridProjector(centers, height.shape, data.metadata["resolution_m_per_px"])
        input_path = out / "inputs" / f"{sample['sample_id']}.npz"
        if not input_path.exists():
            write_npz(input_path, **item, cell_areas=areas, cell_centers=centers,
                      interpolation_footprint=projector.footprint)
        for band in args.bands:
            scene, measurement = backend.scene(vertices, faces, height, item["tx_rc"], data.metadata, band)
            try:
                for seed in args.seeds:
                    for config in CONFIGS:
                        run_case(backend, scene, measurement, projector, len(faces), plan,
                                 band, sample, seed, config, out)
            finally:
                backend.release_scene(scene, measurement)
                del scene, measurement
                gc.collect()
    data.close()
    write_json(out / "completion.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
               "this_invocation_wall_seconds": time.perf_counter() - started, "case_runs": plan["count"]["case_runs"]})
    print(f"Completed. Analyze with: python analyze_ablation.py --run {out}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--bands", nargs="+", choices=("415", "58"), default=["58"])
    p.add_argument("--sample-ids", nargs="+")
    p.add_argument("--terrains", type=int, default=2)
    p.add_argument("--tx-per-terrain", type=int, default=1)
    p.add_argument("--selection-seed", type=int, default=20261026)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--samples-per-tx", type=int, required=True,
                   help="Explicit per-configuration per-seed ray budget; metadata uses 4e9")
    p.add_argument("--rays-per-batch", type=int, default=1000000)
    p.add_argument("--max-depth", type=int, help="Default: metadata max_depth (100); overrides change the physical study")
    p.add_argument("--diagonal", choices=("nw_se", "ne_sw"), default="nw_se")
    p.add_argument("--backend", choices=("cuda", "llvm"), default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
