"""Freeze terrain-disjoint component-dataset splits using geometry only."""

import numpy as np
from scipy.ndimage import map_coordinates

# Already inspected during earlier RT development; keep them out of final external validation.
PREVIOUSLY_INSPECTED_VAL = ("00646", "02408", "02519", "03172", "04229", "04414", "05432", "06213")


def terrain_candidates(data, seed, exclude=()):
    """One existing TX per terrain, chosen before looking at geometry or labels.

    Coarse visibility is ONLY a selection proxy: 8x8 RXs and 33 interior samples
    along each segment with bilinear H interpolation. Actual centroid LOS is
    computed later against the mesh. Time O(terrains * H * W), working memory
    O(H * W), in addition to the read-only heightmap mmap.
    """
    grouped = {}
    for index, row in enumerate(data.rows):
        grouped.setdefault(row["terrain_id"], []).append(index)
    rng = np.random.default_rng(seed)
    candidates = []
    for terrain in sorted(set(grouped) - set(exclude)):
        index = int(rng.choice(grouped[terrain]))
        row = data.rows[index]
        height = np.asarray(data.array("hm")[int(row["hm_row"])], np.float64)
        if height.shape != data.grid or not np.isfinite(height).all():
            raise ValueError(f"Invalid heightmap: {row['sample_id']}")
        r, c = int(row["tx_row"]), int(row["tx_col"])
        if not (0 <= r < height.shape[0] and 0 <= c < height.shape[1]):
            raise ValueError("TX outside terrain")
        low, high = np.quantile(height, [.05, .95])
        gy, gx = np.gradient(height, data.metadata["resolution_m_per_px"])
        rr, cc = np.meshgrid(np.linspace(1, height.shape[0] - 2, 8),
                             np.linspace(1, height.shape[1] - 2, 8), indexing="ij")
        rx_rc = np.stack((rr.ravel(), cc.ravel()))
        tx_rc = np.asarray([r, c])[:, None]
        t = np.linspace(0, 1, 35)[1:-1]
        segment = tx_rc[:, :, None] + (rx_rc - tx_rc)[:, :, None] * t
        terrain_z = map_coordinates(height, segment, order=1, mode="nearest")
        sim = data.metadata["simulation"]
        tx_z = height[r, c] + sim["tx_height_m_agl"]
        rx_z = map_coordinates(height, rx_rc, order=1, mode="nearest") + sim["rx_height_m_agl"]
        line_z = tx_z + (rx_z[:, None] - tx_z) * t
        visible = np.all(line_z > terrain_z, axis=1)
        edge = min(r / (height.shape[0]-1), c / (height.shape[1]-1),
                   1-r / (height.shape[0]-1), 1-c / (height.shape[1]-1))
        candidates.append({"source_split": data.split, "index": index, **row, "parent_id": None,
            "relief_m": float(high-low), "slope_rms": float(np.sqrt(np.mean(gx*gx+gy*gy))),
            "tx_relative_height": float((height[r, c]-low) / max(high-low, 1e-12)),
            "tx_edge": bool(edge < .125), "coarse_los_fraction": float(visible.mean()),
            "available_tx_count": len(grouped[terrain])})
    return candidates


def balanced_take(items, count, rng):
    if not 0 <= count <= len(items):
        raise ValueError("Insufficient distinct terrains for requested split")
    groups = {}
    for item in items:
        groups.setdefault(item["stratum"], []).append(item)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < count:
        for key in keys:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    return chosen


def freeze_selection(train, val, seed, counts=(400, 64, 64)):
    train_ids = {r["terrain_id"] for r in train.rows}
    if train_ids & {r["terrain_id"] for r in val.rows}:
        raise ValueError("Train and val terrain IDs overlap")
    a = terrain_candidates(train, seed)
    b = terrain_candidates(val, seed + 1, PREVIOUSLY_INSPECTED_VAL)
    # Fit bin edges on train geometry only; no target/mask array is accessed.
    relief_edges = np.quantile([r["relief_m"] for r in a], [1/3, 2/3]).tolist()
    height_edge = float(np.median([r["tx_relative_height"] for r in a]))
    for row in a + b:
        row["stratum"] = "_".join(map(str, (
            int(np.searchsorted(relief_edges, row["relief_m"])),
            min(int(row["coarse_los_fraction"] * 3), 2),
            int(row["tx_relative_height"] >= height_edge), int(row["tx_edge"]),
            int(row["available_tx_count"] > 1))))
    rng = np.random.default_rng(seed + 2)
    fitting = balanced_take(a, counts[0], rng)
    used = {r["terrain_id"] for r in fitting}
    internal = balanced_take([r for r in a if r["terrain_id"] not in used], counts[1], rng)
    external = balanced_take(b, counts[2], rng)
    for name, rows in (("fit", fitting), ("internal_val", internal), ("external_val", external)):
        for row in rows:
            row["partition"] = name
    pilot = balanced_take(fitting, min(16, len(fitting)), rng)
    pilot_ids = {r["sample_id"] for r in pilot}
    single_ids = {r["sample_id"] for r in balanced_take(pilot, min(4, len(pilot)), rng)}
    representatives = (balanced_take(fitting, min(24, len(fitting)), rng)
                       + balanced_take(internal, min(4, len(internal)), rng)
                       + balanced_take(external, min(4, len(external)), rng))
    representative_ids = {r["sample_id"] for r in representatives}
    difficult = sorted(representatives, key=lambda r: (r["coarse_los_fraction"], -r["relief_m"]))[:8]
    hard_ids = {r["sample_id"] for r in difficult}
    rows = fitting + internal + external
    for row in rows:
        row["pilot"] = row["sample_id"] in pilot_ids
        row["single_mechanism_control"] = row["sample_id"] in single_ids
        row["representative"] = row["sample_id"] in representative_ids
        row["hard"] = row["sample_id"] in hard_ids
    if len({r["terrain_id"] for r in rows}) != sum(counts):
        raise ValueError("Terrain isolation failed")
    return {"selection_seed": seed, "counts": dict(zip(("fit", "internal_val", "external_val"), counts)),
            "relief_edges_train_m": relief_edges, "tx_height_edge_train": height_edge,
            "stratum_order": ["relief_tertile", "coarse_los_third", "tx_height_half", "edge", "multi_tx"],
            "selection_rule": "balanced_round_robin_over_geometry_strata_one_random_existing_tx",
            "coarse_los_rule": "8x8_RX_33_interior_bilinear_segment_samples_selection_only",
            "parent_id_status": "unknown_not_present_in_source_index_no_parent_isolation_guarantee",
            "excluded_val_terrains": list(PREVIOUSLY_INSPECTED_VAL), "samples": rows}
