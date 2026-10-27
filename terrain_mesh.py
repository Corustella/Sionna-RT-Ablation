"""Deterministic heightfield triangulation and fixed linear-power interpolation."""

import numpy as np
from scipy.spatial import Delaunay

BARYCENTRIC_ZERO_TOLERANCE = 1e-12


def triangulate(height, spacing, diagonal="nw_se"):
    height = np.asarray(height, np.float64)
    if height.ndim != 2 or min(height.shape) < 2 or not np.isfinite(height).all() or spacing <= 0:
        raise ValueError("Finite 2-D terrain and positive spacing are required")
    h, w = height.shape
    rows, cols = np.indices(height.shape)
    vertices = np.column_stack((cols.ravel() * spacing, rows.ravel() * spacing, height.ravel()))
    a = (rows[:-1, :-1] * w + cols[:-1, :-1]).ravel()
    b, c, d = a + 1, a + w, a + w + 1
    if diagonal == "nw_se":
        first, second = np.stack((a, b, d), 1), np.stack((a, d, c), 1)
    elif diagonal == "ne_sw":
        first, second = np.stack((a, b, c), 1), np.stack((b, d, c), 1)
    else:
        raise ValueError("Unknown triangulation diagonal")
    faces = np.stack((first, second), 1).reshape(-1, 3).astype(np.uint32)
    corners = vertices[faces]
    areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) / 2
    return vertices, faces, corners.mean(1), areas


class GridProjector:
    """Fixed centroid-to-grid stencil, including zero gain; no gap filling.

    Stencil depends only on XY, not on values or mechanism-specific positive
    support. Building costs O(M log M); applying it O(HW), memory O(M + HW).
    """

    def __init__(self, centers, shape, spacing):
        self.shape = tuple(shape)
        rows, cols = np.indices(shape)
        query = np.column_stack((cols.ravel() * spacing, rows.ravel() * spacing))
        triangulation = Delaunay(centers[:, :2])
        simplex = triangulation.find_simplex(query)
        self.valid = simplex >= 0
        selected = simplex[self.valid]
        transform = triangulation.transform[selected]
        bary = np.einsum("nij,nj->ni", transform[:, :2], query[self.valid] - transform[:, 2])
        self.weights = np.column_stack((bary, 1 - bary.sum(1)))
        if self.weights.size and self.weights.min() < -1e-8:
            raise ValueError("Unexpected negative interpolation weight")
        self.weights = np.maximum(self.weights, 0)
        # Grid nodes lie on centroid edges. Roundoff in the zero third weight
        # must not turn a zero-hit cell into an arbitrarily weak positive gain.
        # This is a dimensionless geometry tolerance, not a path-gain floor.
        self.weights[self.weights < BARYCENTRIC_ZERO_TOLERANCE] = 0
        self.weights /= self.weights.sum(1, keepdims=True)
        self.vertices = triangulation.simplices[selected]

    @property
    def footprint(self):
        return self.valid.reshape(self.shape)

    def apply(self, cell_gain):
        output = np.full(np.prod(self.shape), np.nan, np.float64)
        output[self.valid] = np.sum(np.asarray(cell_gain)[self.vertices] * self.weights, axis=1)
        return output.reshape(self.shape)
