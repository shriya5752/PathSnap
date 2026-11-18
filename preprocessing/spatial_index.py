"""NumPy nearest-neighbour fallback for the preprocessing shortlist."""

import numpy as np


class SpatialIndex:
    def __init__(self, coordinates):
        self.coordinates = np.asarray(coordinates)
        self.order = np.argsort(self.coordinates[:, 0], kind="stable")
        self.x = self.coordinates[self.order, 0]

    def query(self, points, k):
        """Return the exact k nearest points, breaking distance ties by index."""
        distances = np.empty((len(points), k))
        indices = np.empty((len(points), k), dtype=np.int64)
        for row, point in enumerate(points):
            x = point[0]
            middle = np.searchsorted(self.x, x)
            left = max(0, min(middle - k, len(self.x) - k))
            initial = self.order[left:middle + k]
            delta = self.coordinates[initial] - point
            squared = np.einsum("ij,ij->i", delta, delta)
            radius = np.sqrt(np.partition(squared, k - 1)[k - 1])
            lo = np.searchsorted(self.x, np.nextafter(x - radius, -np.inf))
            hi = np.searchsorted(self.x, np.nextafter(x + radius, np.inf), side="right")
            nearby = self.order[lo:hi]
            delta = self.coordinates[nearby] - point
            squared = np.einsum("ij,ij->i", delta, delta)
            nearest = np.lexsort((nearby, squared))[:k]
            distances[row] = np.sqrt(squared[nearest])
            indices[row] = nearby[nearest]
        return distances, indices
