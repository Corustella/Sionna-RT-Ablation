"""Generate the terrain-disjoint 5.775 GHz same-trace component dataset."""

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from ablation_core import batch_plan, switches, write_json, write_npz
from component_core import (ComponentAccumulator, FAMILIES, GAIN_FIELDS, SCHEMA,
                            geometry_strata, loss_db, select_receivers, validate_components)
from dataset_selection import freeze_selection
from dataset_resume import resume_manifest
from lunar_data import LunarData
from run_ablation import code_version, hardware_info
from terrain_mesh import GridProjector, triangulate


def selected_samples(plan, subset):
    return [row for row in plan["selection"]["samples"] if subset == "all" or row[subset]]


def map_path(out, sample, seed):
    return Path(out) / "maps" / sample["partition"] / sample["sample_id"] / f"seed{seed}.npz"


def make_plan(args, data, selection=None):
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Use at least two distinct seeds; component labels retain each separately")
    if (args.samples_per_tx < 2 or not 1 <= args.rays_per_batch <= 2**31-1
            or args.max_depth < 0 or args.checkpoint_every < 1 or args.log_every < 1):
        raise ValueError("Invalid ray budget, depth or checkpoint/log interval")
    if not 1 <= args.path_receivers <= 2 * (data["train"].grid[0]-1) * (data["train"].grid[1]-1):
        raise ValueError("Invalid path receiver count")
    if (not 1 <= args.path_rx_batch <= args.path_receivers or args.path_samples < 2
            or args.path_samples >= 2**31 or args.path_max_candidates < args.path_rx_batch
            or args.path_max_candidates * max(1, args.max_depth) >= 2**31):
        raise ValueError("Invalid or unbounded PathSolver budget/candidate buffer")
    if args.path_samples * args.path_rx_batch * max(1, args.max_depth) + args.path_rx_batch >= 2**32:
        raise ValueError("PathSolver per-source uint32 candidate counter may overflow; reduce path samples/RX batch")
    for seed in args.seeds:
        batch_plan(args.samples_per_tx, args.rays_per_batch, seed)
    metadata = data["train"].metadata
    if metadata != data["val"].metadata:
        raise ValueError("Train/val metadata mismatch")
    sim = metadata["simulation"]
    if "1.1.0" not in sim["engine"] or sim["propagation_mechanisms"].get("diffraction", False):
        raise ValueError("Requires Sionna RT 1.1.0 metadata without diffraction")
    if selection is None:
        selection = freeze_selection(data["train"], data["val"], args.selection_seed)
    if selection["selection_seed"] != args.selection_seed:
        raise ValueError("Selection seed differs from frozen selection")
    samples = selection["samples"]
    if selection["counts"] != {"fit": 400, "internal_val": 64, "external_val": 64}:
        raise ValueError("Formal selection must contain 400+64+64 distinct terrains")
    if len(samples) != 528 or len({s["terrain_id"] for s in samples}) != 528:
        raise ValueError("Frozen selection is not terrain-disjoint")
    for sample in samples:
        row = data[sample["source_split"]].rows[sample["index"]]
        if any(sample[k] != row[k] for k in row):
            raise ValueError(f"Source row changed: {sample['sample_id']}")
    return {"schema": SCHEMA, "scope": "metadata_based_reconstruction_not_original_generator",
        "metadata": metadata, "data_files": {k: v.provenance(["58"]) for k, v in data.items()},
        "selection": selection, "band": "58", "seeds": args.seeds,
        "samples_per_tx": args.samples_per_tx, "rays_per_batch": args.rays_per_batch,
        "max_depth": args.max_depth, "flags": switches("full"), "rr_depth": -1, "stop_threshold": None,
        "backend": args.backend, "diagonal": args.diagonal,
        "launch_sampling": "disjoint_index_ranges_of_one_global_upstream_fibonacci_lattice",
        "seed_role": "interaction_sampling_only_not_initial_launch_directions",
        "batch_seed_rule": "uint32_SeedSequence([requested_seed,batch_index])",
        "gain_order": list(GAIN_FIELDS), "family_order": list(FAMILIES),
        "native_solver_dtype": "float32", "batch_gain_accumulation_dtype": "float64",
        "native_gain_accumulation_dtype": "float64",
        "hit_counter_dtype": "uint64", "mesh_coordinates": "x=column*spacing,y=row*spacing,z=H",
        "scene_geometry": "open_heightfield_no_sidewalls_or_bottom_actual_FP32_vertices_saved",
        "measurement": "same_mesh_translated_up_in_z_by_metadata_rx_height",
        "path_gain_definition": "dimensionless_cell_area_mean_incoherent_ray_power_upstream_weights",
        "grid_projection": "fixed_XY_centroid_Delaunay_linear_gain_no_flip_no_fill",
        "loss_definition": "-10*log10(gain); zero=+inf; outside_footprint=NaN",
        "zero_status": "unresolved_or_zero_not_evidence_of_no_physical_path",
        "centroid_los": "terrain_segment_ray_test_endpoint_shortening_1e-4_m_not_cell_area_visibility",
        "original_generator_unknowns": ["original_mesh", "coordinate_flip_convention", "adaptive_resampling"],
        "official_base_ray_budget_matched": args.samples_per_tx == int(sim["samples_per_tx"]),
        "checkpoint_every_batches": args.checkpoint_every, "log_every_batches": args.log_every,
        "paths": {"scene_subset": "representative", "receivers_per_scene": args.path_receivers,
            "rx_batch": args.path_rx_batch, "samples_per_src_per_rx_batch": args.path_samples,
            "max_candidates_per_src": args.path_max_candidates, "max_depth": args.max_depth,
            "synthetic_array": True, "rx_antenna": "1x1 iso VH orientation=(0,0,0)",
            "layout": "native_valid_path_records; a[path,rx_pol,tx_pol]; interactions[path,depth]; rx_index_is_local_to_chunk",
            "polarization_rule": "keep_V_and_H_complex_coefficients_separate; diagnostic_power=sum_abs_a_squared_over_polarizations_and_paths",
            "meaning": "separate_point_PathSolver_diagnostic_not_weighted_RadioMap_ray_records",
            "seed_rule": "uint32_SeedSequence([requested_seed,rx_batch_index])",
            "candidate_cap_policy": "fail_before_shrink_if_count_ge_cap; no_silent_truncation"}}


def save_selection(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_map(path, samples=None, cells=None):
    with np.load(path, allow_pickle=False) as saved:
        gain = np.stack([saved[key] for key in ("G_all",) + FAMILIES])
        hits, positive = saved["hit_count"], saved["positive_hit_count"]
        if int(saved["schema"]) != SCHEMA or (samples is not None and int(saved["samples"]) != samples):
            raise ValueError(f"Schema/ray count mismatch: {path}")
        if cells is not None and gain.shape != (6, cells):
            raise ValueError(f"Cell count mismatch: {path}")
        validate_components(gain, hits, positive)
        if (tuple(saved["gain_order"]) != GAIN_FIELDS or tuple(saved["family_order"]) != FAMILIES
                or not np.array_equal(saved["G_other"], gain[3:].sum(axis=0))):
            raise ValueError(f"Component field order or G_other mismatch: {path}")
        return gain


def run_map(backend, scene, measurement, projector, cells, plan, sample, seed, out, float32_samples=0):
    final = map_path(out, sample, seed)
    if final.exists():
        read_map(final, plan["samples_per_tx"], cells)
        print(f"skip map {sample['sample_id']} seed={seed}", flush=True)
        return
    checkpoint = final.with_name(final.stem + ".progress.npz")
    if float32_samples and not checkpoint.exists():
        raise ValueError(f"Recorded FP32 checkpoint is missing; restore it before resume: {checkpoint}")
    accumulator = ComponentAccumulator(cells)
    schedule = batch_plan(plan["samples_per_tx"], plan["rays_per_batch"], seed)
    completed, elapsed = 0, 0.0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            completed, elapsed = int(saved["completed_batches"]), float(saved["elapsed_seconds"])
            accumulator.samples = int(saved["samples"])
            accumulator.weighted = saved["weighted_gain"].copy()
            accumulator.hits = saved["hit_count"].copy()
            accumulator.positive_hits = saved["positive_hit_count"].copy()
            if "float32_accumulation_samples" in saved and int(saved["float32_accumulation_samples"]) != float32_samples:
                raise ValueError(f"Checkpoint accumulation provenance mismatch: {checkpoint}")
        if (not 0 <= completed <= len(schedule) or accumulator.samples <= 0
                or accumulator.samples != sum(b["samples"] for b in schedule[:completed])
                or accumulator.weighted.shape != (6, cells) or not 0 <= float32_samples <= accumulator.samples
                or not np.isfinite(elapsed) or elapsed < 0):
            raise ValueError(f"Invalid progress file: {checkpoint}")
        validate_components(accumulator.mean(), accumulator.hits, accumulator.positive_hits)
    for i in range(completed, len(schedule)):
        batch = schedule[i]
        started = time.perf_counter()
        gain, hits, positive = backend.solve_components(scene, measurement, batch["samples"], batch["seed"],
            plan["max_depth"], total_samples=plan["samples_per_tx"], sample_offset=batch["offset"])
        accumulator.add(gain, hits, positive, batch["samples"])
        elapsed += time.perf_counter() - started
        if (i+1) % plan["checkpoint_every_batches"] == 0 or i+1 == len(schedule):
            write_npz(checkpoint, weighted_gain=accumulator.weighted, hit_count=accumulator.hits,
                positive_hit_count=accumulator.positive_hits, samples=np.int64(accumulator.samples),
                completed_batches=np.int64(i+1), elapsed_seconds=elapsed,
                float32_accumulation_samples=np.int64(float32_samples))
        if i == completed or (i+1) % plan["log_every_batches"] == 0 or i+1 == len(schedule):
            print(f"{sample['sample_id']} seed={seed}: batch {i+1}/{len(schedule)}, "
                  f"rays={accumulator.samples}, compute={elapsed:.2f}s", flush=True)
    gain = accumulator.mean()
    closure = validate_components(gain, accumulator.hits, accumulator.positive_hits)
    all_gain = np.concatenate((gain, gain[3:].sum(axis=0, keepdims=True)))
    grid_gain = np.stack([projector.apply(g) for g in all_gain])
    arrays = dict(zip(GAIN_FIELDS, all_gain))
    write_npz(final, **arrays, schema=np.int64(SCHEMA), gain_order=np.asarray(GAIN_FIELDS),
        family_order=np.asarray(FAMILIES), grid_gain=grid_gain, grid_pathloss_db=loss_db(grid_gain),
        interpolation_footprint=projector.footprint, hit_count=accumulator.hits,
        positive_hit_count=accumulator.positive_hits, unresolved_or_zero=all_gain == 0,
        samples=np.int64(accumulator.samples), seed=np.int64(seed), elapsed_seconds=elapsed,
        float32_accumulation_samples=np.int64(float32_samples),
        float64_accumulation_samples=np.int64(accumulator.samples-float32_samples),
        closure_max_relative_error=closure["max_relative_error"])
    # Only our own redundant progress file is removed, after the atomic final write.
    checkpoint.unlink(missing_ok=True)


def run_paths(backend, scene, geometry, plan, sample, seed, out):
    settings = plan["paths"]
    cells = geometry["path_rx_cell_indices"]
    schedule = range(0, len(cells), settings["rx_batch"])
    directory = out / "paths" / sample["sample_id"] / f"seed{seed}"
    for chunk_index, offset in enumerate(schedule):
        chosen = cells[offset:offset+settings["rx_batch"]]
        final = directory / f"rx{offset:04d}.npz"
        if final.exists():
            with np.load(final, allow_pickle=False) as saved:
                if (not np.array_equal(saved["rx_cell_indices"], chosen)
                        or int(saved["samples_per_src"]) != settings["samples_per_src_per_rx_batch"]
                        or bool(saved["candidate_cap_reached"])):
                    raise ValueError(f"Invalid path diagnostic file: {final}")
            continue
        actual_seed = int(np.random.SeedSequence([seed, chunk_index]).generate_state(1)[0])
        positions = geometry["cell_centers"][chosen].astype(np.float32)
        print(f"paths {sample['sample_id']} seed={seed}: starting RX {offset+1}-{offset+len(chosen)}/{len(cells)}, "
              f"samples/src={settings['samples_per_src_per_rx_batch']}, depth={settings['max_depth']}; "
              "first call may compile GPU kernels", flush=True)
        started = time.perf_counter()
        paths = backend.point_paths(scene, positions, actual_seed,
            settings["samples_per_src_per_rx_batch"], settings["max_depth"], settings["max_candidates_per_src"])
        write_npz(final, **paths, schema=np.int64(SCHEMA), seed=np.int64(seed), solver_seed=np.int64(actual_seed),
            rx_cell_indices=chosen, rx_xyz=positions, tx_xyz=geometry["tx_xyz"],
            rx_region=geometry["region"][chosen], rx_distance_bin=geometry["distance_bin"][chosen],
            samples_per_src=np.int64(settings["samples_per_src_per_rx_batch"]),
            elapsed_seconds=time.perf_counter()-started)
        print(f"paths {sample['sample_id']} seed={seed}: RX {offset+len(chosen)}/{len(cells)}", flush=True)


def save_geometry(backend, scene, measurement, item, plan, sample, out, projector):
    geometry_path = out / "geometry" / f"{sample['sample_id']}.npz"
    if (geometry_path.exists() and (out / "reference" / f"{sample['sample_id']}.npz").exists()
            and geometry_path.with_suffix(".json").exists()):
        with np.load(geometry_path, allow_pickle=False) as saved:
            geometry = {key: saved[key].copy() for key in saved.files}
        return geometry
    geometry = backend.geometry(scene, measurement)
    visible = backend.centroid_los(scene, geometry["cell_centers"])
    region, distance_bin, distance = geometry_strata(visible, geometry["cell_centers"],
                                                    geometry["tx_xyz"], geometry["measurement_faces"])
    geometry.update(centroid_los=visible, region=region, distance_bin=distance_bin, distance_m=distance,
        height_m=item["height_m"], tx_rc=item["tx_rc"], interpolation_footprint=projector.footprint,
        path_rx_cell_indices=select_receivers(region, distance_bin,
            min(plan["paths"]["receivers_per_scene"], len(visible)),
            plan["selection"]["selection_seed"] + sample["index"]))
    write_npz(geometry_path, **geometry)
    write_npz(out / "reference" / f"{sample['sample_id']}.npz",
              official_pathloss_db=item["target_58"], official_valid_mask=item["mask_58"])
    write_json(out / "geometry" / f"{sample['sample_id']}.json", {
        "sample": sample, "tx_antenna": plan["metadata"]["simulation"]["tx_antenna"],
        "tx_power_dbm": plan["metadata"]["simulation"]["tx_power_dbm"],
        "material": plan["metadata"]["simulation"]["material"],
        "transform": "identity", "source_height_dtype": "float32_stored_losslessly_as_float64",
        "mesh_vertex_dtype": "float32", "face_dtype": "uint32", "centers_areas_dtype": "float64",
        "region_labels": {"0": "visible_interior", "1": "centroid_LOS_boundary", "2": "shadow_interior"},
        "path_object_id_rule": "process_local_ID_to_mesh_name_mapping_saved_in_each_path_chunk",
        "path_vertex_normals": "recover_face_normals_from_saved_vertices_and_primitive_indices"})
    return geometry


def dataset_status(out, plan):
    rows, seeds = plan["selection"]["samples"], plan["seeds"]
    maps = sum(map_path(out, row, seed).exists() for row in rows for seed in seeds)
    expected_maps = len(rows) * len(seeds)
    settings = plan["paths"]
    chunks = len(range(0, settings["receivers_per_scene"], settings["rx_batch"]))
    expected_paths = sum(r["representative"] for r in rows) * len(seeds) * chunks
    paths = sum((out / "paths" / row["sample_id"] / f"seed{seed}" / f"rx{offset:04d}.npz").exists()
                for row in rows if row["representative"] for seed in seeds
                for offset in range(0, settings["receivers_per_scene"], settings["rx_batch"]))
    return {"map_files": maps, "expected_map_files": expected_maps,
            "path_chunk_files": paths, "expected_path_chunk_files": expected_paths,
            "all_requested_files_present": maps == expected_maps and paths == expected_paths}


def main(args):
    data = {split: LunarData(args.data_root, split) for split in ("train", "val")}
    try:
        out = Path(args.out).expanduser().resolve()
        selection = None
        if args.selection_from:
            source = json.loads(Path(args.selection_from).read_text(encoding="utf-8"))["experiment"]
            if source["metadata"] != data["train"].metadata or source["data_files"] != {k: v.provenance(["58"]) for k, v in data.items()}:
                raise ValueError("Frozen selection uses different source data")
            selection = source["selection"]
        elif args.resume and (out / "manifest.json").exists():
            selection = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["experiment"]["selection"]
        if selection is None:
            print("Selecting 528 terrains using geometry only; scanning train/val heightmaps once.", flush=True)
        plan = make_plan(args, data, selection)
        chosen = selected_samples(plan, args.subset)
        report = {"partition_counts": plan["selection"]["counts"], "execution_subset": args.subset,
            "selected_scenes_this_invocation": len(chosen), "seeds": plan["seeds"],
            "rays_per_scene_per_seed": plan["samples_per_tx"], "mode": args.mode,
            "map_trace_budget_for_requested_subset": len(chosen)*len(plan["seeds"]) if args.mode != "paths" else 0,
            "map_ray_budget_for_requested_subset": len(chosen)*len(plan["seeds"])*plan["samples_per_tx"] if args.mode != "paths" else 0,
            "budget_note": "planned_subset_budget_includes_files_that_resume_will_skip; not_an_actual_launch_counter",
            "path_settings": plan["paths"], "out": str(out), "scope": plan["scope"]}
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            print("Dry run: no RT import, simulation, or files written.", flush=True)
            return
        if args.upgrade_fp32_accumulation and not args.resume:
            raise ValueError("--upgrade-fp32-accumulation requires --resume")
        if args.upgrade_fp32_accumulation and not (out / "manifest.json").exists():
            raise ValueError("FP32 upgrade requires an existing legacy manifest; check --out")
        version = code_version()
        if version["dirty"] or version["commit"] is None:
            raise ValueError("Dataset generation requires a committed, clean checkout for verifiable resume")
        print("Initializing Sionna/Mitsuba backend (first call may compile GPU kernels)...", flush=True)
        from component_backend import ComponentBackend
        backend = ComponentBackend(args.backend)
        print("RT backend ready; checking output manifest...", flush=True)
        if out.exists():
            if not args.resume:
                raise FileExistsError(f"Output exists: {out}; use identical settings with --resume")
            manifest = resume_manifest(out, plan, version, backend.versions, args.upgrade_fp32_accumulation)
        else:
            out.mkdir(parents=True)
            write_json(out / "manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": plan, "code": version, "runtime": backend.versions, "command": sys.argv,
                "hardware": hardware_info(), "python": sys.version, "platform": platform.platform()})
            save_selection(out / "selection.csv", plan["selection"]["samples"])
            manifest = {}
        invocation_start = time.perf_counter()
        projector = None
        for sample in chosen:
            print(f"Preparing scene {sample['sample_id']} ({sample['partition']})...", flush=True)
            item = data[sample["source_split"]].sample(sample["index"], ["58"])
            vertices, faces, centers, _ = triangulate(item["height_m"],
                plan["metadata"]["resolution_m_per_px"], plan["diagonal"])
            if projector is None:
                projector = GridProjector(centers, item["height_m"].shape, plan["metadata"]["resolution_m_per_px"])
            scene, measurement = backend.scene(vertices, faces, item["height_m"], item["tx_rc"], plan["metadata"], "58")
            try:
                geometry = save_geometry(backend, scene, measurement, item, plan, sample, out, projector)
                for seed in plan["seeds"]:
                    if args.mode != "paths":
                        boundary = manifest.get("accumulation_upgrade", {}).get("float32_samples_by_map", {})
                        key = map_path(out, sample, seed).relative_to(out).as_posix()
                        run_map(backend, scene, measurement, projector, len(faces), plan, sample, seed, out,
                                float32_samples=boundary.get(key, 0))
                    if args.mode != "maps" and sample["representative"]:
                        run_paths(backend, scene, geometry, plan, sample, seed, out)
            finally:
                backend.release_scene(scene, measurement)
                del scene, measurement
                gc.collect()
        status = dataset_status(out, plan)
        write_json(out / "status.json", status)
        write_json(out / "invocations" / f"{time.time_ns()}.json", {
            "finished_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
            "subset": args.subset, "mode": args.mode, "wall_seconds": time.perf_counter()-invocation_start,
            "status": status})
        print(json.dumps(status, indent=2), flush=True)
    finally:
        for source in data.values():
            source.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "dataset"))
    p.add_argument("--selection-from", help="Frozen manifest.json; reuse its 528 terrains for diagnostics")
    p.add_argument("--selection-seed", type=int, default=20261029)
    p.add_argument("--subset", choices=("all", "pilot", "representative", "hard"), default="all")
    p.add_argument("--mode", choices=("all", "maps", "paths"), default="all")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--samples-per-tx", type=int, default=4000000000)
    p.add_argument("--rays-per-batch", type=int, default=1000000)
    p.add_argument("--max-depth", type=int, default=100)
    p.add_argument("--diagonal", choices=("nw_se", "ne_sw"), default="nw_se")
    p.add_argument("--backend", choices=("cuda", "llvm"), default="cuda")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--path-receivers", type=int, default=256)
    p.add_argument("--path-rx-batch", type=int, default=32)
    p.add_argument("--path-samples", type=int, default=100000,
                   help="Separate PathSolver samples_per_src per RX batch; not the RadioMap 4e9 budget")
    p.add_argument("--path-max-candidates", type=int, default=250000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--upgrade-fp32-accumulation", action="store_true",
                   help="Explicitly upgrade compatible c0408be checkpoints; preserve old FP32 sums and record the boundary")
    p.add_argument("--dry-run", action="store_true")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
