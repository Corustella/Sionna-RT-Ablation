"""Check declared 5.775 GHz settings and export filled labels without re-tracing.

Unknown interpolation/averaging domains and the adaptive stopping rule are not
inferred from labels. The mean domain must be supplied explicitly. Source maps,
their component closure, and generation checkpoints are never modified.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from ablation_core import switches, write_json, write_npz
from analyze_component_dataset import write_csv
from generate_dataset import map_path, selected_samples
from run_ablation import code_version


def check_settings(plan, runtime):
    metadata = plan["metadata"]
    sim = metadata["simulation"]
    expected = {
        "band": (plan["band"], "58"), "frequency_hz": (metadata["bands_hz"]["58"], 5775000000.),
        "grid": (metadata["grid"], [256, 256]),
        "spacing_m": (metadata["resolution_m_per_px"], 1.),
        "tx_height_m": (sim["tx_height_m_agl"], 3.), "rx_height_m": (sim["rx_height_m_agl"], 1.),
        "tx_power_dbm": (sim["tx_power_dbm"], 20.),
        "tx_antenna": (sim["tx_antenna"], "PlanarArray 1x1, pattern='iso', polarization='V', one transmitter per tile"),
        "rx_probe": (sim["rx_antenna"], "none modeled; isotropic probe at each sample point"),
        "bandwidth_hz": (sim["bandwidth_hz"]["58"], 80000000.),
        "max_depth": (plan["max_depth"], 100), "samples_per_seed": (plan["samples_per_tx"], 4000000000),
        "flags": (plan["flags"], switches("full")),
        "diffraction": (sim["propagation_mechanisms"]["diffraction"], False),
        "sionna_rt_version": (runtime["sionna-rt"], "1.1.0"),
        "fill_db": (sim["gap_filling"]["static_fill_pathloss_db"], 145.),
    }
    material = sim["material"]
    for key, value in (("relative_permittivity", 3.411), ("thickness_m", 10.),
                       ("scattering_coefficient", .632), ("xpd_coefficient", .1),
                       ("scattering_pattern", "lambertian")):
        expected["material_" + key] = (material[key], value)
    expected["conductivity_s_per_m"] = (material["conductivity_s_per_m"]["58"], .009324)
    failed = [name for name, (actual, value) in expected.items() if actual != value]
    if failed:
        raise ValueError("Declared settings differ from the official metadata profile: " + ", ".join(failed))
    return {name: actual for name, (actual, _) in expected.items()}


def check_geometry(geometry, metadata):
    terrain, measurement = geometry["terrain_vertices"], geometry["measurement_vertices"]
    if terrain.shape != measurement.shape or not np.array_equal(geometry["terrain_faces"], geometry["measurement_faces"]):
        raise ValueError("Measurement must be a copy of the terrain mesh")
    expected = terrain.astype(np.float64).copy()
    expected[:, 2] += metadata["simulation"]["rx_height_m_agl"]
    # The uploaded meshes store FP32 positions. Compare to that exact rounding.
    if not np.array_equal(measurement, expected.astype(np.float32)):
        raise ValueError("Saved measurement mesh is not the +1m terrain copy")
    row, col = map(int, geometry["tx_rc"])
    spacing = metadata["resolution_m_per_px"]
    position = np.asarray([col * spacing, row * spacing,
        geometry["height_m"][row, col] + metadata["simulation"]["tx_height_m_agl"]], np.float32)
    if not np.array_equal(geometry["tx_xyz"], position):
        raise ValueError("Saved TX differs from the current pixel-coordinate/+3m convention")
    return {"measurement_is_translated_copy": True, "tx_height_and_current_coordinate_convention_checked": True,
            "original_world_coordinate_convention_verified": False}


def pathloss(gain):
    result = np.full(gain.shape, np.nan, np.float64)
    good = np.isfinite(gain) & (gain > 0)
    result[good] = -10 * np.log10(gain[good])
    return result


def combine(gains, domain):
    """Interpolate first (saved maps), then ignore unresolved seeds per pixel.

    Domain is an explicit assumption, not a claim about the original script.
    Complexity O(SHW); the caller holds S saved maps, no RT state.
    """
    gains = np.asarray(gains, np.float64)
    good = np.isfinite(gains) & (gains > 0)
    count = good.sum(axis=0)
    if domain == "gain":
        values = gains
    elif domain == "db":
        values = pathloss(gains)
    else:
        raise ValueError("Unknown mean domain")
    mean = np.divide(np.where(good, values, 0).sum(axis=0), count,
                     out=np.full(gains.shape[1:], np.nan), where=count > 0)
    return (pathloss(mean) if domain == "gain" else mean), count


def filled(pl, dtype=np.float16):
    resolved = np.isfinite(pl)
    label = np.where(resolved, pl, 145.).astype(dtype)
    if not np.isfinite(label).all():
        raise ValueError("Finite path loss cannot be represented in the output dtype")
    return label, resolved


def statistics(pl, target, official_mask):
    mask = np.asarray(official_mask, bool)
    if pl.shape != target.shape or pl.shape != mask.shape or not np.isfinite(target[mask]).all():
        raise ValueError("Invalid label/mask shape or nonfinite official valid label")
    prediction, resolved = filled(pl)
    overlap = mask & resolved
    error = prediction.astype(np.float64)[mask] - target[mask]
    overlap_error = pl[overlap] - target[overlap]
    return {"official_valid_count": int(mask.sum()), "resolved_official_valid_count": int(overlap.sum()),
            "filled_official_valid_count": int((mask & ~resolved).sum()),
            "coverage": int(overlap.sum()) / int(mask.sum()) if mask.any() else None,
            "masked_sse_db": float(np.square(error).sum()),
            "masked_rmse_db": float(np.sqrt(np.mean(error**2))) if error.size else None,
            "overlap_sse_db_before_quantization": float(np.square(overlap_error).sum()),
            "overlap_rmse_db_before_quantization": float(np.sqrt(np.mean(overlap_error**2))) if overlap_error.size else None}


def pooled(rows):
    result = []
    for partition, variant in sorted({(r["partition"], r["variant"]) for r in rows}):
        selected = [r for r in rows if r["partition"] == partition and r["variant"] == variant]
        n = sum(r["official_valid_count"] for r in selected)
        k = sum(r["resolved_official_valid_count"] for r in selected)
        sse = sum(r["masked_sse_db"] for r in selected)
        overlap_sse = sum(r["overlap_sse_db_before_quantization"] for r in selected)
        result.append({"partition": partition, "variant": variant, "maps": len(selected),
            "official_valid_count": n, "resolved_official_valid_count": k, "coverage": k/n if n else None,
            "masked_sse_db": sse, "masked_rmse_db": float(np.sqrt(sse/n)) if n else None,
            "overlap_rmse_db_before_quantization": float(np.sqrt(overlap_sse/k)) if k else None})
    return result


def main(args):
    root = Path(args.dataset).expanduser().resolve()
    out, results = Path(args.out).expanduser().resolve(), Path(args.results).expanduser().resolve()
    if out.exists() or results.exists():
        raise FileExistsError("Use new export/results directories; source and prior exports are preserved")
    if out == root or out in root.parents or results == root or results in root.parents:
        raise ValueError("Do not write over the source dataset or its parents")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    plan = manifest["experiment"]
    checks = check_settings(plan, manifest["runtime"])
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(plan["seeds"]):
        raise ValueError("Use distinct seeds from the frozen dataset plan")
    samples = [s for s in selected_samples(plan, args.subset) if not args.sample_ids or s["sample_id"] in args.sample_ids]
    rows, missing, exported, geometry_checks, input_maps = [], [], [], [], []
    for sample in samples:
        sid = sample["sample_id"]
        paths = [map_path(root, sample, seed) for seed in args.seeds]
        absent = [str(p) for p in paths if not p.exists()]
        if absent:
            missing.extend(absent)
            continue  # Never silently change the requested number of seeds.
        with np.load(root / "geometry" / f"{sid}.npz", allow_pickle=False) as saved:
            geometry_checks.append({"sample_id": sid, **check_geometry(saved, plan["metadata"])})
        with np.load(root / "reference" / f"{sid}.npz", allow_pickle=False) as saved:
            target, mask = saved["official_pathloss_db"].copy(), saved["official_valid_mask"].astype(bool)
        gains = []
        for seed, path in zip(args.seeds, paths):
            with np.load(path, allow_pickle=False) as saved:
                if int(saved["samples"]) != 4000000000:
                    raise ValueError(f"Expected a completed 4B map: {path}")
                grid = saved["grid_gain"][0].copy()
                if np.isinf(grid).any() or (grid[np.isfinite(grid)] < 0).any():
                    raise ValueError(f"Invalid gain, not an unresolved zero: {path}")
                gains.append(grid)
                input_maps.append({"path": str(path), "seed": seed, "samples": int(saved["samples"]),
                    "float32_accumulation_samples": int(saved["float32_accumulation_samples"]) if "float32_accumulation_samples" in saved else None})
        baseline = pathloss(np.mean(gains, axis=0))
        combined, count = combine(gains, args.mean_domain)
        variants = {f"seed{seed}": pathloss(g) for seed, g in zip(args.seeds, gains)}
        variants.update(baseline_gain_mean_including_zeros=baseline, aligned_nanmean=combined)
        for variant, pl in variants.items():
            rows.append({"sample_id": sid, "partition": sample["partition"], "variant": variant,
                         **statistics(pl, target, mask)})
        label, resolved = filled(combined)
        path = out / sample["partition"] / f"{sid}.npz"
        write_npz(path, pathloss_db=label, resolved_mask=resolved, unfilled_pathloss_db=combined,
                  contributing_seeds=count.astype(np.uint16), requested_seeds=np.asarray(args.seeds),
                  mean_domain=np.asarray(args.mean_domain))
        exported.append(str(path))
        print(f"Exported {sid}; completed seeds={args.seeds}", flush=True)
    local_note = root / "local_test_provenance.json"
    summary = {"source": str(root), "exports": exported, "exported_scenes": len(exported),
        "requested_scenes": len(samples), "requested_seeds": args.seeds, "missing_maps": missing,
        "declared_settings_checked": checks, "saved_geometry_checked": geometry_checks, "input_maps": input_maps,
        "source_code": manifest.get("code"), "export_code": code_version(), "command": sys.argv,
        "local_test_provenance": json.loads(local_note.read_text(encoding="utf-8")) if local_note.exists() else None,
        "mean_domain_assumption": args.mean_domain, "interpolation": "unchanged_saved_grid_linear_gain",
        "known_operations": ["per-pixel finite-only seed mean", "positive_pathloss_db", "unresolved_fill_145_db", "float16_storage"],
        "not_reproduced": ["original_mesh_and_coordinate_context", "original_interpolation_value_domain",
            "original_seed_mean_value_domain", "adaptive_15_initial_60_refinement_attempts_and_diminishing_returns_cutoff",
            "original_per_ray_random_stream_and_accumulation_order"],
        "interpretation": "Partial alignment only. Mean domain explicitly assumed, not established by metadata. "
            "Fixed requested seeds, not the official adaptive stopping rule. resolved_mask is reconstruction support, "
            "NEVER the official scoring mask. Main RMSE scores every official valid pixel, including filled outputs. "
            "All variants in each scene have identical official denominators and float16 output format.",
        "pooled": pooled(rows)}
    write_json(out / "manifest.json", summary)
    write_csv(results / "maps.csv", rows)
    write_csv(results / "pooled.csv", summary["pooled"])
    write_json(results / "summary.json", summary)
    print(json.dumps({"exported_scenes": len(exported), "missing_maps": len(missing), "results": str(results)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--out", default="dataset/aligned_labels")
    parser.add_argument("--results", default="results/official_known_alignment")
    parser.add_argument("--subset", choices=("all", "pilot", "representative", "hard"), default="all")
    parser.add_argument("--sample-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--mean-domain", required=True, choices=("gain", "db"),
                        help="Required explicit assumption; official mean domain is not disclosed")
    options = parser.parse_args()
    if len(options.seeds) < 2:
        parser.error("Use at least two seeds")
    main(options)
