"""Full-budget switch ablations against the saved full component-dataset trace."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

from ablation_core import (GainAccumulator, batch_plan, compare_gains, switches,
                           write_json, write_npz)
from analyze_component_dataset import write_csv
from generate_dataset import map_path, read_map, selected_samples
from run_ablation import code_version, hardware_info
from terrain_mesh import GridProjector

ONLY_FLAGS = {
    "specular_only": dict(los=False, specular_reflection=True, diffuse_reflection=False, refraction=False),
    "diffuse_only": dict(los=False, specular_reflection=False, diffuse_reflection=True, refraction=False),
    "refraction_only": dict(los=False, specular_reflection=False, diffuse_reflection=False, refraction=True),
}


def cases(sample, configs, controls):
    result = list(configs)
    if controls and sample["pilot"]:
        result.append("los_only")
        if sample["single_mechanism_control"]:
            result.extend(ONLY_FLAGS)
    return list(dict.fromkeys(result))


def run_case(backend, scene, measurement, projector, cells, plan, seed, config, final):
    if final.exists():
        with np.load(final, allow_pickle=False) as saved:
            if int(saved["samples"]) != plan["samples_per_tx"] or saved["cell_gain"].shape != (cells,):
                raise ValueError(f"Existing ablation file mismatch: {final}")
        return
    accumulator, completed, elapsed = GainAccumulator((cells,)), 0, 0.0
    schedule = batch_plan(plan["samples_per_tx"], plan["rays_per_batch"], seed)
    checkpoint = final.with_name(final.stem + ".progress.npz")
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            accumulator.weighted, accumulator.samples = saved["weighted_gain"].copy(), int(saved["samples"])
            completed, elapsed = int(saved["completed_batches"]), float(saved["elapsed_seconds"])
        if (not 0 <= completed <= len(schedule)
                or accumulator.samples != sum(b["samples"] for b in schedule[:completed])
                or accumulator.weighted.shape != (cells,)
                or not np.isfinite(accumulator.weighted).all() or (accumulator.weighted < 0).any()
                or not np.isfinite(elapsed) or elapsed < 0):
            raise ValueError(f"Invalid ablation progress: {checkpoint}")
    flags = ONLY_FLAGS[config] if config in ONLY_FLAGS else switches(config)
    depth = 0 if config == "los_only" else plan["max_depth"]
    for i in range(completed, len(schedule)):
        batch = schedule[i]
        started = time.perf_counter()
        gain = backend.solve(scene, measurement, batch["samples"], batch["seed"], depth, flags,
                             total_samples=plan["samples_per_tx"], sample_offset=batch["offset"])
        accumulator.add(gain, batch["samples"])
        elapsed += time.perf_counter() - started
        if (i+1) % plan["checkpoint_every_batches"] == 0 or i+1 == len(schedule):
            write_npz(checkpoint, weighted_gain=accumulator.weighted, samples=np.int64(accumulator.samples),
                      completed_batches=np.int64(i+1), elapsed_seconds=elapsed)
        if i == completed or (i+1) % plan["log_every_batches"] == 0 or i+1 == len(schedule):
            print(f"{final.parent.parent.name} seed={seed} {config}: "
                  f"batch {i+1}/{len(schedule)}, compute={elapsed:.2f}s", flush=True)
    gain = accumulator.mean()
    write_npz(final, cell_gain=gain, grid_gain=projector.apply(gain), samples=np.int64(accumulator.samples),
              seed=np.int64(seed), elapsed_seconds=elapsed)
    checkpoint.unlink(missing_ok=True)


def analyze(root, out, results, plan, samples, configs, controls):
    rows, los_checks = [], []
    for sample in samples:
        with np.load(root / "geometry" / f"{sample['sample_id']}.npz", allow_pickle=False) as geometry:
            areas = geometry["cell_areas"].copy()
        for seed in plan["seeds"]:
            full_path = map_path(root, sample, seed)
            with np.load(full_path, allow_pickle=False) as full:
                for config in cases(sample, configs, controls):
                    path = out / "maps" / sample["sample_id"] / f"seed{seed}" / f"{config}.npz"
                    if not path.exists():
                        continue
                    with np.load(path, allow_pickle=False) as ablated:
                        for domain, a, b, mask, weights in (
                            ("native", full["G_all"], ablated["cell_gain"], np.ones(len(areas), bool), areas),
                            ("grid", full["grid_gain"][0], ablated["grid_gain"], full["interpolation_footprint"], None)):
                            statistics = compare_gains(a, b, mask, weights)
                            count, sse = statistics["both_positive_count"], statistics["delta_db_squared_sum"]
                            rows.append({"sample_id": sample["sample_id"], "partition": sample["partition"],
                                "seed": seed, "config": config, "domain": domain,
                                **{k: v for k, v in statistics.items() if "quantiles" not in k},
                                "rmse_db_on_both_positive": float(np.sqrt(sse/count)) if count else None})
                        if config == "los_only":
                            a, b = full["G_LOS"], ablated["cell_gain"]
                            relative = np.divide(np.abs(a-b), np.maximum(a, b),
                                                 out=np.zeros_like(a), where=np.maximum(a, b) > 0)
                            los_checks.append({"sample_id": sample["sample_id"], "seed": seed,
                                "max_relative_error_full_trace_LOS_vs_LOS_only": float(relative.max(initial=0)),
                                "within_1e_minus4_relative_tolerance": bool((relative <= 1e-4).all())})
    pooled = []
    for domain in ("native", "grid"):
        for config in sorted({r["config"] for r in rows}):
            for seed in (None, *plan["seeds"]):
                selected = [r for r in rows if r["domain"] == domain and r["config"] == config
                            and (seed is None or r["seed"] == seed)]
                count = sum(r["both_positive_count"] for r in selected)
                sse = sum(r["delta_db_squared_sum"] for r in selected)
                full_power = sum(r["weighted_full_gain_sum"] for r in selected)
                ablated_power = sum(r["weighted_ablated_gain_sum"] for r in selected)
                pooled.append({"domain": domain, "config": config, "seed": seed, "map_count": len(selected),
                    "both_positive_count": count, "sse_db": sse,
                    "rmse_db_on_both_positive": float(np.sqrt(sse/count)) if count else None,
                    "lost_to_zero_count": sum(r["lost_to_zero_count"] for r in selected),
                    "gained_from_zero_count": sum(r["gained_from_zero_count"] for r in selected),
                    "relative_gain_removed": (full_power-ablated_power)/full_power if full_power else None})
    write_csv(results / "per_map.csv", rows)
    write_csv(results / "summary.csv", pooled)
    write_csv(results / "los_classification_check.csv", los_checks)
    write_json(results / "summary.json", {"full_dataset": str(root), "ablation_run": str(out),
        "summary": pooled, "meaning": "Switch-off marginal effect, not mutually exclusive family shares. "
        "Pooled SSE/count on common positive support; lost/gained zeros reported separately. "
        "Native RMSE counts triangles equally; native power integrates cell areas. "
        "Grid uses fixed footprint including official-invalid pixels: simulator-to-simulator comparison."})
    print(f"Wrote {results / 'summary.csv'}", flush=True)


def main(args):
    root, out = args.dataset.expanduser().resolve(), args.out.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan = manifest["experiment"]
    samples = selected_samples(plan, args.subset)
    if args.analyze_only:
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        request = previous["request"]
        # A downloaded dataset can live at a different absolute path. Match its
        # immutable manifest rather than requiring the original server path.
        if request["dataset_manifest"] != manifest or request["subset"] != args.subset:
            raise ValueError("Analysis dataset manifest/subset differs from the saved ablation request")
        analyze(root, out, args.results, plan, samples, request["configs"], request["controls"])
        return
    request = {"dataset": str(root), "dataset_manifest": manifest, "subset": args.subset,
               "configs": args.configs, "controls": args.controls}
    case_count = sum(len(cases(s, args.configs, args.controls)) for s in samples) * len(plan["seeds"])
    print(json.dumps({"scenes": len(samples), "independent_trace_budget": case_count,
                      "rays_per_trace": plan["samples_per_tx"],
                      "extra_ray_budget": case_count * plan["samples_per_tx"], "out": str(out)}, indent=2), flush=True)
    if args.dry_run:
        return
    for sample in samples:
        for seed in plan["seeds"]:
            read_map(map_path(root, sample, seed), plan["samples_per_tx"])
    from rt_backend import RTBackend
    backend = RTBackend(plan["backend"])
    version = code_version()
    if version["dirty"] or version != manifest["code"] or backend.versions != manifest["runtime"]:
        raise ValueError("Ablation requires the same clean code and RT runtime as the full dataset")
    if out.exists():
        if not args.resume:
            raise FileExistsError("Ablation output exists; use identical settings with --resume")
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if previous["request"] != request or previous["code"] != version:
            raise ValueError("Ablation resume settings/code mismatch")
    else:
        write_json(out / "manifest.json", {"request": request, "code": version, "runtime": backend.versions,
            "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv, "hardware": hardware_info()})
    for sample in samples:
        with np.load(root / "geometry" / f"{sample['sample_id']}.npz", allow_pickle=False) as saved:
            geometry = {key: saved[key].copy() for key in saved.files}
        projector = GridProjector(geometry["cell_centers"], geometry["height_m"].shape,
                                  plan["metadata"]["resolution_m_per_px"])
        scene, measurement = backend.scene(geometry["terrain_vertices"], geometry["terrain_faces"],
            geometry["height_m"], geometry["tx_rc"], plan["metadata"], "58")
        try:
            for seed in plan["seeds"]:
                for config in cases(sample, args.configs, args.controls):
                    final = out / "maps" / sample["sample_id"] / f"seed{seed}" / f"{config}.npz"
                    run_case(backend, scene, measurement, projector, len(geometry["cell_areas"]), plan, seed, config, final)
        finally:
            backend.release_scene(scene, measurement)
            del scene, measurement
            gc.collect()
    analyze(root, out, args.results, plan, samples, args.configs, args.controls)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("dataset"))
    p.add_argument("--out", type=Path, default=Path("runs/dataset_ablation_4b"))
    p.add_argument("--results", type=Path, default=Path("results/dataset_ablation_4b"))
    p.add_argument("--subset", choices=("all", "pilot", "representative", "hard"), default="pilot")
    p.add_argument("--configs", nargs="+", choices=("no_los", "no_specular", "no_diffuse", "no_refraction"),
                   default=["no_los", "no_specular", "no_diffuse", "no_refraction"])
    p.add_argument("--controls", action="store_true", help="Add LOS-only on 16 pilots and three single-mechanism runs on four pilots")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    main(args)
