"""Mechanism switches and statistics; independent of the RT runtime."""

import json
from pathlib import Path

import numpy as np


CONFIGS = {
    "full": (True, True, True, True),
    "los_only": (True, False, False, False),
    "no_los": (False, True, True, True),
    "no_specular": (True, False, True, True),
    "no_diffuse": (True, True, False, True),
    "no_refraction": (True, True, True, False),
}
FLAG_NAMES = ("los", "specular_reflection", "diffuse_reflection", "refraction")


def switches(name):
    return dict(zip(FLAG_NAMES, CONFIGS[name]))


def batch_plan(total, batch_size, seed):
    """Partition ONE global Fibonacci lattice; pair interaction RNG across cases."""
    if total < 2 or batch_size < 1 or not 0 <= seed < 2**32:
        raise ValueError("At least two total rays, positive batch size and a uint32 seed are required")
    result = []
    for i, start in enumerate(range(0, total, batch_size)):
        derived = int(np.random.SeedSequence([seed, i]).generate_state(1)[0])
        result.append({"samples": min(batch_size, total - start), "offset": start, "seed": derived})
    return result


class GainAccumulator:
    """Ray-count-weighted mean, including zero-hit estimates. O(number of cells)."""

    def __init__(self, shape):
        self.weighted = np.zeros(shape, np.float64)
        self.samples = 0

    def add(self, gain, samples):
        gain = np.asarray(gain, np.float64)
        if gain.shape != self.weighted.shape or samples < 1:
            raise ValueError("Invalid gain shape or sample count")
        if not np.isfinite(gain).all() or (gain < 0).any():
            raise ValueError("RT returned negative or nonfinite path gain")
        self.weighted += gain * samples
        self.samples += samples

    def mean(self):
        if not self.samples:
            raise ValueError("No completed batches")
        return self.weighted / self.samples


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def compare_gains(full, ablated, domain, weights=None):
    """Signed leave-one-out effects, without clipping or masking away lost cells.

    Linear fractions overlap for mixed paths. dB statistics use the common
    positive support, whose denominator and losses are reported separately.
    Runtime O(N log N) for quantiles, memory O(N).
    """
    full, ablated = np.asarray(full, np.float64), np.asarray(ablated, np.float64)
    domain = np.asarray(domain, bool)
    if full.shape != ablated.shape or full.shape != domain.shape:
        raise ValueError("Comparison arrays must have the same shape")
    weights = np.ones_like(full) if weights is None else np.asarray(weights, np.float64)
    if weights.shape != full.shape:
        raise ValueError("Weight shape mismatch")
    if (not np.isfinite(full[domain]).all() or not np.isfinite(ablated[domain]).all()
            or (full[domain] < 0).any() or (ablated[domain] < 0).any()
            or not np.isfinite(weights[domain]).all() or (weights[domain] <= 0).any()):
        raise ValueError("Invalid gain or weights inside the fixed comparison domain")
    positive_full = domain & (full > 0)
    positive_ablated = domain & (ablated > 0)
    common = positive_full & positive_ablated
    total_full = float(np.sum(full[domain] * weights[domain]))
    total_ablated = float(np.sum(ablated[domain] * weights[domain]))
    delta_db = 10 * (np.log10(full[common]) - np.log10(ablated[common]))
    count = int(domain.sum())
    return {
        "domain_count": count,
        "domain_weight": float(weights[domain].sum()),
        "full_positive_count": int(positive_full.sum()),
        "ablated_positive_count": int(positive_ablated.sum()),
        "both_positive_count": int(common.sum()),
        "lost_to_zero_count": int((positive_full & ~positive_ablated).sum()),
        "gained_from_zero_count": int((positive_ablated & ~positive_full).sum()),
        "both_zero_count": int((domain & ~positive_full & ~positive_ablated).sum()),
        "ablated_exceeds_full_count": int((domain & (ablated > full)).sum()),
        "weighted_full_gain_sum": total_full,
        "weighted_ablated_gain_sum": total_ablated,
        "relative_gain_removed": safe_ratio(total_full - total_ablated, total_full),
        "full_positive_fraction": safe_ratio(int(positive_full.sum()), count),
        "lost_fraction_of_full_positive": safe_ratio(
            int((positive_full & ~positive_ablated).sum()), int(positive_full.sum())),
        "delta_pathloss_db_mean_on_both_positive": float(delta_db.mean()) if delta_db.size else None,
        "delta_pathloss_db_quantiles_on_both_positive": (
            dict(zip(("p10", "p50", "p90"), map(float, np.quantile(delta_db, [.1, .5, .9]))))
            if delta_db.size else None),
        "delta_db_sum": float(delta_db.sum()),
        "delta_db_squared_sum": float(np.square(delta_db).sum()),
    }


def label_diagnostic(gain, target_db, official_mask, footprint):
    """Not an official challenge score: unresolved pixels remain unresolved."""
    domain = np.asarray(official_mask, bool)
    footprint = np.asarray(footprint, bool)
    supported = domain & footprint
    overlap = supported & np.isfinite(gain) & (gain > 0)
    if not np.isfinite(target_db[domain]).all():
        raise ValueError("Nonfinite target inside the official validity mask")
    error = -10 * np.log10(gain[overlap]) - target_db[overlap]
    sse = float(np.square(error).sum())
    count = int(overlap.sum())
    return {
        "official_valid_count": int(domain.sum()),
        "valid_inside_interpolation_footprint_count": int(supported.sum()),
        "positive_overlap_count": count,
        "valid_but_zero_count": int((supported & (gain == 0)).sum()),
        "valid_outside_interpolation_footprint_count": int((domain & ~footprint).sum()),
        "sse_db_on_positive_overlap": sse,
        "rmse_db_on_positive_overlap": float(np.sqrt(sse / count)) if count else None,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
