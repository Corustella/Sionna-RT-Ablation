"""Audit component closure, coverage, seed agreement and optional budget agreement."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from ablation_core import label_diagnostic, write_json
from component_core import FAMILIES, GAIN_FIELDS, validate_components
from generate_dataset import dataset_status, map_path, read_map, selected_samples


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(a, b, domain, a_total, b_total, reliable):
    common = domain & (a > 0) & (b > 0)
    delta = 10 * (np.log10(a[common]) - np.log10(b[common]))
    trusted = common & reliable
    trusted_delta = 10 * (np.log10(a[trusted]) - np.log10(b[trusted]))
    share_domain = domain & (a_total > 0) & (b_total > 0)
    share_delta = a[share_domain] / a_total[share_domain] - b[share_domain] / b_total[share_domain]
    return {"domain_count": int(domain.sum()), "both_positive_count": int(common.sum()),
        "a_positive_b_zero_count": int((domain & (a > 0) & (b == 0)).sum()),
        "a_zero_b_positive_count": int((domain & (a == 0) & (b > 0)).sum()),
        "log_power_sse_db": float(np.square(delta).sum()),
        "log_power_absolute_sum_db": float(np.abs(delta).sum()),
        "reliable_count": int(trusted.sum()), "reliable_log_power_sse_db": float(np.square(trusted_delta).sum()),
        "share_count": int(share_domain.sum()), "share_absolute_sum": float(np.abs(share_delta).sum()),
        "log_power_rmse_db_on_both_positive": float(np.sqrt(np.mean(delta**2))) if delta.size else None}


def compare_maps(a, b, geometry, min_hits, identity):
    ag, bg = np.stack([a[key] for key in GAIN_FIELDS[:6]]), np.stack([b[key] for key in GAIN_FIELDS[:6]])
    ap, bp = a["positive_hit_count"], b["positive_hit_count"]
    reliable_a = np.concatenate((ap.sum(axis=0, keepdims=True), ap)) >= min_hits
    reliable_b = np.concatenate((bp.sum(axis=0, keepdims=True), bp)) >= min_hits
    rows = []
    for region in (-1, 0, 1, 2):
        domain = np.ones(len(ag[0]), bool) if region == -1 else geometry["region"] == region
        for index, key in enumerate(GAIN_FIELDS[:6]):
            rows.append({**identity, "region": region, "component": key,
                **paired_statistics(ag[index], bg[index], domain, ag[0], bg[0],
                                    reliable_a[index] & reliable_b[index])})
    return rows


def summarize_pairs(rows):
    groups = {}
    for row in rows:
        key = (row["comparison"], row["partition"], row["region"], row["component"])
        group = groups.setdefault(key, {"paired_maps": 0})
        group["paired_maps"] += 1
        for field in ("domain_count", "both_positive_count", "a_positive_b_zero_count", "a_zero_b_positive_count",
                      "log_power_sse_db", "log_power_absolute_sum_db", "reliable_count",
                      "reliable_log_power_sse_db", "share_count", "share_absolute_sum"):
            group[field] = group.get(field, 0) + row[field]
    result = []
    for key, group in sorted(groups.items()):
        n, nr, nq = group["both_positive_count"], group["reliable_count"], group["share_count"]
        result.append({**dict(zip(("comparison", "partition", "region", "component"), key)), **group,
            "pooled_log_power_rmse_db": float(np.sqrt(group["log_power_sse_db"]/n)) if n else None,
            "pooled_reliable_log_power_rmse_db": float(np.sqrt(group["reliable_log_power_sse_db"]/nr)) if nr else None,
            "pooled_share_mae": group["share_absolute_sum"]/nq if nq else None})
    return result


def audit_paths(root, plan, samples):
    """Validate compact path-record axes and expose RXs with no found path."""
    rows, missing = [], []
    settings = plan["paths"]
    for sample in samples:
        if not sample["representative"]:
            continue
        for seed in plan["seeds"]:
            for offset in range(0, settings["receivers_per_scene"], settings["rx_batch"]):
                path = root / "paths" / sample["sample_id"] / f"seed{seed}" / f"rx{offset:04d}.npz"
                if not path.exists():
                    missing.append(str(path))
                    continue
                with np.load(path, allow_pickle=False) as saved:
                    valid, a, b = saved["valid"], saved["a_real"], saved["a_imag"]
                    rx_index = saved["rx_index"].astype(np.int64)
                    rx_count, count = len(saved["rx_cell_indices"]), len(valid)
                    interactions = saved["interactions"]
                    if (valid.shape != (count,) or a.shape != (count, 2, 1) or b.shape != a.shape
                            or rx_index.shape != (count,) or (rx_index < 0).any() or (rx_index >= rx_count).any()
                            or bool(saved["candidate_cap_reached"]) or not valid.all()
                            or interactions.shape != (count, max(1, settings["max_depth"]))
                            or not np.isin(interactions, [0, 1, 2, 4]).all()
                            or not np.isfinite(a).all() or not np.isfinite(b).all()
                            or not np.isfinite(saved["delay_s"]).all() or (saved["delay_s"] < 0).any()):
                        raise ValueError(f"Invalid compact path records: {path}")
                    intensity = (a.astype(np.float64)**2 + b.astype(np.float64)**2).sum(axis=(1, 2))
                    per_rx_count = np.bincount(rx_index, minlength=rx_count)
                    per_rx_power = np.bincount(rx_index, weights=intensity, minlength=rx_count)
                    for index in range(rx_count):
                        rows.append({"sample_id": sample["sample_id"], "partition": sample["partition"],
                            "seed": seed, "cell_index": int(saved["rx_cell_indices"][index]),
                            "region": int(saved["rx_region"][index]), "distance_bin": int(saved["rx_distance_bin"][index]),
                            "valid_path_count": int(per_rx_count[index]),
                            "point_incoherent_dual_pol_gain": float(per_rx_power[index]),
                            "candidate_count_chunk": int(saved["candidate_count_before_rejection"]),
                            "samples_per_src_chunk": int(saved["samples_per_src"])})
    return rows, missing


def main(args):
    root, out = Path(args.dataset).expanduser().resolve(), Path(args.out).expanduser().resolve()
    plan = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["experiment"]
    other = Path(args.other_run).expanduser().resolve() if args.other_run else None
    other_plan = None
    if other:
        other_plan = json.loads((other / "manifest.json").read_text(encoding="utf-8"))["experiment"]
        for key in ("selection", "metadata", "data_files", "max_depth", "flags", "diagonal", "gain_order"):
            if other_plan[key] != plan[key]:
                raise ValueError(f"Budget comparison mismatch: {key}")
    map_rows, pair_rows, missing = [], [], []
    selected = selected_samples(plan, args.subset)
    for sample in selected:
        sid = sample["sample_id"]
        identity = {"sample_id": sid, "terrain_id": sample["terrain_id"], "partition": sample["partition"]}
        present = [seed for seed in plan["seeds"] if map_path(root, sample, seed).exists()]
        missing.extend(str(map_path(root, sample, seed)) for seed in plan["seeds"] if seed not in present)
        if not present:
            continue
        with np.load(root / "geometry" / f"{sid}.npz", allow_pickle=False) as geo_file:
            geometry = {key: geo_file[key].copy() for key in ("cell_areas", "region")}
        with np.load(root / "reference" / f"{sid}.npz", allow_pickle=False) as ref:
            target, mask = ref["official_pathloss_db"].copy(), ref["official_valid_mask"].copy()
        for seed in present:
            path = map_path(root, sample, seed)
            read_map(path, plan["samples_per_tx"], len(geometry["cell_areas"]))
            with np.load(path, allow_pickle=False) as result:
                gain = np.stack([result[key] for key in GAIN_FIELDS[:6]])
                closure = validate_components(gain, result["hit_count"], result["positive_hit_count"])
                row = {**identity, "seed": seed, "samples": int(result["samples"]), **closure,
                    "float32_accumulation_samples": int(result["float32_accumulation_samples"]) if "float32_accumulation_samples" in result else int(result["samples"]),
                    "float64_accumulation_samples": int(result["float64_accumulation_samples"]) if "float64_accumulation_samples" in result else 0,
                    "elapsed_seconds": float(result["elapsed_seconds"]),
                    **label_diagnostic(result["grid_gain"][0], target, mask, result["interpolation_footprint"])}
                integrated = np.sum(gain * geometry["cell_areas"], axis=1)
                for k, field in enumerate(GAIN_FIELDS[:6]):
                    row[field + "_positive_cells"] = int((gain[k] > 0).sum())
                    row[field + "_area_integral_m2"] = float(integrated[k])
                    row[field + "_area_integrated_share"] = float(integrated[k]/integrated[0]) if integrated[0] > 0 else None
                map_rows.append(row)
        for first, second in itertools.combinations(present, 2):
            with np.load(map_path(root, sample, first), allow_pickle=False) as a, np.load(map_path(root, sample, second), allow_pickle=False) as b:
                pair_rows.extend(compare_maps(a, b, geometry, args.min_hits,
                    {**identity, "comparison": "seed", "a": str(first), "b": str(second)}))
        if other:
            for seed in sorted(set(present) & set(other_plan["seeds"])):
                path = map_path(other, sample, seed)
                if not path.exists():
                    missing.append(str(path))
                    continue
                read_map(path, other_plan["samples_per_tx"], len(geometry["cell_areas"]))
                with np.load(map_path(root, sample, seed), allow_pickle=False) as a, np.load(path, allow_pickle=False) as b:
                    pair_rows.extend(compare_maps(a, b, geometry, args.min_hits,
                        {**identity, "comparison": "budget", "a": str(plan["samples_per_tx"]),
                         "b": str(other_plan["samples_per_tx"]), "seed": seed}))
    label_groups = []
    for partition in sorted({r["partition"] for r in map_rows}):
        for seed in plan["seeds"]:
            rows = [r for r in map_rows if r["partition"] == partition and r["seed"] == seed]
            count = sum(r["positive_overlap_count"] for r in rows)
            sse = sum(r["sse_db_on_positive_overlap"] for r in rows)
            valid = sum(r["official_valid_count"] for r in rows)
            label_groups.append({"partition": partition, "seed": seed, "map_count": len(rows),
                "official_valid_count": valid, "positive_overlap_count": count,
                "positive_overlap_fraction_of_official_valid": count/valid if valid else None,
                "sse_db_on_positive_overlap": sse,
                "rmse_db_on_positive_overlap": float(np.sqrt(sse/count)) if count else None})
    path_rows, missing_paths = audit_paths(root, plan, selected)
    summary = {"status": dataset_status(root, plan), "analyzed_maps": len(map_rows),
        "missing_files": missing, "min_positive_hits_for_reliable_log_error": args.min_hits,
        "audited_point_receivers": len(path_rows), "point_receivers_without_found_path": sum(r["valid_path_count"] == 0 for r in path_rows),
        "missing_path_chunks": missing_paths,
        "label_diagnostic": label_groups, "pair_statistics": summarize_pairs(pair_rows),
        "interpretation": "Pooled SSE/count; missing/zero excluded from log error but counted explicitly. "
                          "Official-label comparison is overlap-only reconstruction diagnostic, not a challenge score. "
                          "Seed comparisons share the initial fixed Fibonacci directions. "
                          "Region 0=visible interior,1=boundary,2=shadow; -1=all. "
                          "Path diagnostics are separate point solves; file presence is not convergence certification."}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "maps.csv", map_rows)
    write_csv(out / "paired_maps.csv", pair_rows)
    write_csv(out / "paired_summary.csv", summary["pair_statistics"])
    write_csv(out / "point_receivers.csv", path_rows)
    write_json(out / "summary.json", summary)
    print(json.dumps({"analyzed_maps": len(map_rows), "missing_count": len(missing),
                      "status": summary["status"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--out", default="results/component_dataset")
    p.add_argument("--other-run", help="Second component run with identical frozen terrain selection, e.g. 2e9 rays")
    p.add_argument("--subset", choices=("all", "pilot", "representative", "hard"), default="all")
    p.add_argument("--min-hits", type=int, default=8)
    options = p.parse_args()
    if options.min_hits < 1:
        p.error("--min-hits must be positive")
    main(options)
