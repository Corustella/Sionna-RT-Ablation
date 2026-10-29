"""Component schema and validation, without importing the RT runtime."""

import numpy as np

FAMILIES = ("G_LOS", "G_spec", "G_diff_family", "G_trans_family", "G_diff_trans")
GAIN_FIELDS = ("G_all",) + FAMILIES + ("G_other",)
SCHEMA = 1
CLOSURE_RTOL = 5e-5


def validate_components(gain, hits=None, positive_hits=None):
    """Check same-trace closure without a gain floor, including tiny components."""
    gain = np.asarray(gain)
    if gain.ndim != 2 or gain.shape[0] != 6:
        raise ValueError("Expected [G_all, five families] x native cells")
    if not np.isfinite(gain).all() or (gain < 0).any():
        raise ValueError("Negative or nonfinite component gain")
    total = gain[1:].sum(axis=0, dtype=np.float64)
    residual = np.abs(gain[0] - total)
    scale = np.maximum(gain[0], total)
    positive = scale > 0
    relative = np.divide(residual, scale, out=np.zeros_like(scale), where=positive)
    if (relative > CLOSURE_RTOL).any():
        raise ValueError(f"Same-trace closure failed: max relative error {relative.max():.6g}")
    if hits is not None:
        hits, positive_hits = np.asarray(hits), np.asarray(positive_hits)
        if (hits.shape != gain[1:].shape or positive_hits.shape != hits.shape
                or hits.dtype != np.uint64 or positive_hits.dtype != np.uint64
                or (positive_hits > hits).any()):
            raise ValueError("Invalid per-family uint64 hit counters")
        if ((gain[1:] > 0) & (positive_hits == 0)).any():
            raise ValueError("Positive component without a positive-weight hit")
    return {"max_relative_error": float(relative.max(initial=0)),
            "max_absolute_error": float(residual.max(initial=0)),
            "relative_tolerance": CLOSURE_RTOL}


class ComponentAccumulator:
    """FP64 ray-weighted gain sums and uint64 hit sums; O(native cells) memory."""

    def __init__(self, cells):
        self.weighted = np.zeros((6, cells), np.float64)
        self.hits = np.zeros((5, cells), np.uint64)
        self.positive_hits = np.zeros((5, cells), np.uint64)
        self.samples = 0

    def add(self, gain, hits, positive_hits, samples):
        validate_components(gain, hits, positive_hits)
        if gain.shape != self.weighted.shape or samples < 1:
            raise ValueError("Component accumulator shape or ray count mismatch")
        self.weighted += gain * samples
        self.hits += hits
        self.positive_hits += positive_hits
        self.samples += samples

    def mean(self):
        if not self.samples:
            raise ValueError("No completed batches")
        gain = self.weighted / self.samples
        validate_components(gain, self.hits, self.positive_hits)
        return gain


def loss_db(gain):
    """Zero gain -> +inf; outside interpolation footprint -> NaN; no filling."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -10 * np.log10(gain)


def geometry_strata(visible, centers, tx_xyz, faces):
    """Actual face-adjacency LOS boundary and four equal-width distance bands.

    Region 0=visible interior, 1=visibility boundary (on either side), 2=shadow
    interior. A centroid ray is a geometry diagnostic, not area visibility.
    Building adjacency costs O(M log M) time / O(M) memory.
    """
    visible = np.asarray(visible, bool)
    faces = np.asarray(faces)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    owners = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    a, b = owners[:-1][shared], owners[1:][shared]
    cross = visible[a] != visible[b]
    boundary = np.zeros(len(faces), bool)
    boundary[a[cross]] = True
    boundary[b[cross]] = True
    region = np.where(boundary, 1, np.where(visible, 0, 2)).astype(np.uint8)
    distance = np.linalg.norm(centers - np.asarray(tx_xyz), axis=1)
    bins = np.minimum((distance / max(float(distance.max()), 1e-12) * 4).astype(int), 3)
    return region, bins.astype(np.uint8), distance


def select_receivers(region, distance_bin, count, seed):
    """Balanced sampling over existing geometry/distance strata, without labels."""
    if not 1 <= count <= len(region):
        raise ValueError("RX subset must fit inside the native cells")
    rng = np.random.default_rng(seed)
    groups = [list(rng.permutation(np.flatnonzero(region * 4 + distance_bin == k))) for k in range(12)]
    chosen = []
    while len(chosen) < count:
        for group in groups:
            if group and len(chosen) < count:
                chosen.append(group.pop())
    return np.asarray(sorted(chosen), np.int64)
