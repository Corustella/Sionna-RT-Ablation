"""Analyze complete paired ablations in linear power, coverage, and dB."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ablation_core import CONFIGS, compare_gains, label_diagnostic, safe_ratio, write_json


def pooled_statistics(records):
    """Pool additive numerators/counts, never average per-map percentages or RMSE."""
    additive = ("domain_count", "domain_weight", "full_positive_count", "ablated_positive_count",
                "both_positive_count", "lost_to_zero_count", "gained_from_zero_count", "both_zero_count",
                "ablated_exceeds_full_count", "weighted_full_gain_sum", "weighted_ablated_gain_sum",
                "delta_db_sum", "delta_db_squared_sum")
    out = {key: sum(row[key] for row in records) for key in additive}
    out["relative_gain_removed"] = safe_ratio(out["weighted_full_gain_sum"] - out["weighted_ablated_gain_sum"], out["weighted_full_gain_sum"])
    out["lost_fraction_of_full_positive"] = safe_ratio(out["lost_to_zero_count"], out["full_positive_count"])
    out["delta_pathloss_db_mean_on_both_positive"] = safe_ratio(out["delta_db_sum"], out["both_positive_count"])
    return out


def analyze(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    plan = manifest["experiment"]
    comparisons, labels, closure = [], [], []
    for sample in plan["samples"]:
        sid = sample["sample_id"]
        with np.load(run / "inputs" / f"{sid}.npz", allow_pickle=False) as saved:
            inputs = {key: saved[key] for key in saved.files}
        for band in plan["bands"]:
            by_seed = []
            for seed in plan["seeds"]:
                cases = {}
                for case in CONFIGS:
                    path = run / "maps" / sid / band / f"seed{seed}" / f"{case}.npz"
                    if not path.exists():
                        raise FileNotFoundError(f"Incomplete paired experiment: {path}. Resume the runner first.")
                    with np.load(path, allow_pickle=False) as saved:
                        if int(saved["samples"]) != plan["samples_per_tx"]:
                            raise ValueError(f"Unequal ray budgets: {path}")
                        cases[case] = {key: np.asarray(saved[key]) for key in ("cell_gain", "grid_gain")}
                by_seed.append(cases)
            # Average linear estimates including zeros before any logarithm.
            averaged = {case: {key: np.mean([s[case][key] for s in by_seed], axis=0)
                               for key in ("cell_gain", "grid_gain")} for case in CONFIGS}
            for seed, cases in list(zip(plan["seeds"], by_seed)) + [("mean", averaged)]:
                identity = {"sample_id": sid, "terrain_id": sample["terrain_id"], "band": band, "seed": seed}
                full = cases["full"]
                mesh_domain = np.ones_like(full["cell_gain"], bool)
                grid_domain = inputs[f"mask_{band}"] & inputs["interpolation_footprint"]
                for case in CONFIGS:
                    labels.append({**identity, "configuration": case, **label_diagnostic(
                        cases[case]["grid_gain"], inputs[f"target_{band}"], inputs[f"mask_{band}"], inputs["interpolation_footprint"])})
                    if case == "full":
                        continue
                    for domain, field, mask, weights in (
                        ("mesh_area", "cell_gain", mesh_domain, inputs["cell_areas"]),
                        ("official_valid_grid", "grid_gain", grid_domain, None),
                    ):
                        comparisons.append({**identity, "configuration": case, "domain": domain,
                            **compare_gains(full[field], cases[case][field], mask, weights)})
                residual = full["cell_gain"] - cases["los_only"]["cell_gain"] - cases["no_los"]["cell_gain"]
                scale = float(np.sum(full["cell_gain"] * inputs["cell_areas"]))
                closure.append({**identity,
                    "signed_relative_full_minus_los_minus_nonlos": safe_ratio(float(np.sum(residual * inputs["cell_areas"])), scale),
                    "absolute_relative_full_minus_los_minus_nonlos": safe_ratio(float(np.sum(np.abs(residual) * inputs["cell_areas"])), scale)})
    return manifest, comparisons, labels, closure


def main(args):
    run = Path(args.run).resolve()
    out = Path(args.out).resolve() if args.out else run / "analysis"
    if out.exists():
        raise FileExistsError(f"Analysis output already exists; use a new --out: {out}")
    manifest, comparisons, labels, closure = analyze(run)
    mean_rows = [r for r in comparisons if r["seed"] == "mean"]
    summary = []
    terrain_rows = []
    for band in manifest["experiment"]["bands"]:
        for domain in ("mesh_area", "official_valid_grid"):
            for case in CONFIGS:
                if case == "full":
                    continue
                rows = [r for r in mean_rows if r["band"] == band and r["domain"] == domain and r["configuration"] == case]
                terrain_ids = sorted({r["terrain_id"] for r in rows})
                terrain_values = []
                for terrain in terrain_ids:
                    stats = pooled_statistics([r for r in rows if r["terrain_id"] == terrain])
                    terrain_rows.append({"band": band, "domain": domain, "configuration": case, "terrain_id": terrain, **stats})
                    if stats["relative_gain_removed"] is not None:
                        terrain_values.append(stats["relative_gain_removed"])
                seed_values = {}
                for seed in manifest["experiment"]["seeds"]:
                    selected = [r for r in comparisons if r["band"] == band and r["domain"] == domain and r["configuration"] == case and r["seed"] == seed]
                    seed_values[str(seed)] = pooled_statistics(selected)
                effects = [v["relative_gain_removed"] for v in seed_values.values()
                           if v["relative_gain_removed"] is not None]
                summary.append({"band": band, "domain": domain, "configuration": case,
                                "samples": len(rows), "terrains": len(terrain_ids),
                                **pooled_statistics(rows), "per_seed": seed_values,
                                "rt_seed_effect_min": min(effects) if effects else None,
                                "rt_seed_effect_max": max(effects) if effects else None,
                                "equal_terrain_mean_relative_gain_removed": float(np.mean(terrain_values)) if terrain_values else None})
    label_summary = []
    for band in manifest["experiment"]["bands"]:
        for case in CONFIGS:
            rows = [r for r in labels if r["seed"] == "mean" and r["band"] == band and r["configuration"] == case]
            count = sum(r["positive_overlap_count"] for r in rows)
            sse = sum(r["sse_db_on_positive_overlap"] for r in rows)
            label_summary.append({"band": band, "configuration": case,
                "official_valid_count": sum(r["official_valid_count"] for r in rows),
                "positive_overlap_count": count,
                "rmse_db_on_positive_overlap_not_challenge_score": float(np.sqrt(sse / count)) if count else None})
    out.mkdir(parents=True)
    write_json(out / "summary.json", {"scope": manifest["experiment"]["scope"],
        "interpretation": "Signed marginal mechanism removal effects; not disjoint energy fractions. Zero means zero finite-sample estimate, not proven absence of a physical path.",
        "contribution_summary": summary, "label_diagnostics": label_summary,
        "closure_diagnostics": closure})
    for name, rows in (("comparisons", comparisons), ("label_diagnostics", labels), ("terrain_comparisons", terrain_rows)):
        write_json(out / f"{name}.json", rows)
    columns = ["band", "domain", "configuration", "samples", "terrains", "relative_gain_removed",
               "rt_seed_effect_min", "rt_seed_effect_max",
               "lost_fraction_of_full_positive", "delta_pathloss_db_mean_on_both_positive",
               "both_positive_count", "ablated_exceeds_full_count", "equal_terrain_mean_relative_gain_removed"]
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        if row["domain"] == "official_valid_grid":
            fraction = row["relative_gain_removed"]
            printed = "undefined" if fraction is None else f"{fraction:.4%}"
            seeds = ", ".join(f"seed{k}={v['relative_gain_removed']:.4%}"
                              if v["relative_gain_removed"] is not None else f"seed{k}=undefined"
                              for k, v in row["per_seed"].items())
            print(f"band={row['band']} {row['configuration']}: signed gain removed={printed}, "
                  f"positive overlap={row['both_positive_count']}; {seeds}")
    print(f"Saved {out}. Inspect coverage, seed variation and closure before interpreting contributions.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", help="New directory; default RUN/analysis")
    main(p.parse_args())
