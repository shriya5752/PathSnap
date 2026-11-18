from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray

def read(path, columns, integer=False):
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] != columns or not np.isfinite(a).all():
        raise ValueError(f'Invalid shape or nonfinite values: {path}')
    if integer:
        if not np.equal(a, np.floor(a)).all():
            raise ValueError(f'Noninteger ID: {path}')
        a = a.astype(np.int64)
    return a

def load_trip(directory):
    p = Path(directory)
    track = read(p / f'{p.name}.track', 3)
    nodes = read(p / f'{p.name}.nodes', 2)
    arcs = read(p / f'{p.name}.arcs', 2, True)
    for a in (track, nodes):
        if (abs(a[:, 0]) > 180).any() or (abs(a[:, 1]) > 90).any():
            raise ValueError(f'Invalid coordinates: {p}')
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError(f'Invalid node ID: {p}')
    return Trip(p.name, track, nodes, arcs)

def load_reference(directory, n_arcs):
    p = Path(directory)
    route = read(p / f'{p.name}.route', 1, True).ravel()
    if (route < 0).any() or (route >= n_arcs).any():
        raise ValueError(f'Invalid reference arc ID: {p}')
    return route
