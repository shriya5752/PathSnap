import numpy as np
import pandas as pd
try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError as error:
    if error.name != 'scipy':
        raise
    from .spatial_index import SpatialIndex as cKDTree

FEATURE_COLUMNS = ['distance_m', 'heading_diff_deg', 'speed_mps', 'segment_length_m', 'time_gap_s', 'projection_fraction', 'heading_valid', 'time_valid']

def project(trip):
    """Local equirectangular metres, centered at trip mean latitude/longitude."""
    origin = trip.track[:, :2].mean(axis=0)
    scale = np.array([np.cos(np.deg2rad(origin[1])), 1.0]) * (6371008.8 * np.pi / 180)
    return (trip.track[:, :2] - origin) * scale, (trip.nodes - origin) * scale

def segment_distance(point, start, vector):
    length2 = (vector * vector).sum(axis=-1)
    fraction = np.clip(np.divide(((point-start)*vector).sum(axis=-1), length2, out=np.zeros_like(length2), where=length2>0), 0, 1)
    distance = np.linalg.norm(point-start-fraction[...,None]*vector, axis=-1)
    return distance, fraction

def candidate_features(trip, k=10, shortlist=64):
    """Reference-free candidate generation; approximate midpoint/end-node shortlist."""
    if k < 1 or shortlist < k:
        raise ValueError('Require shortlist >= k >= 1')
    points, nodes = project(trip)
    a, b = nodes[trip.arcs[:,0]], nodes[trip.arcs[:,1]]
    v = b-a
    lengths = np.linalg.norm(v, axis=1)
    tree = cKDTree((a+b)/2)
    _, near = tree.query(points, k=min(shortlist,len(a)))
    near = near.reshape(len(points),-1)
    # Include arcs incident to nearby endpoints, mitigating long-segment misses.
    _, closest_nodes = cKDTree(nodes).query(points, k=min(8,len(nodes)))
    closest_nodes = closest_nodes.reshape(len(points),-1)
    endpoints = trip.arcs.ravel()
    order = np.argsort(endpoints, kind='stable')
    offsets = np.searchsorted(endpoints[order], np.arange(len(nodes)+1))
    delta = np.diff(points, axis=0, prepend=points[:1])
    gap = np.diff(trip.track[:,2], prepend=trip.track[0,2])
    movement = np.linalg.norm(delta,axis=1)
    speed = np.divide(movement,gap,out=np.zeros_like(gap),where=gap>0)
    heading = np.arctan2(delta[:,1],delta[:,0])
    rows = []
    for i,p in enumerate(points):
        incident = [order[offsets[n]:offsets[n+1]]//2 for n in closest_nodes[i]]
        ids = np.unique(np.concatenate([near[i],*incident]))
        dist, frac = segment_distance(p,a[ids],v[ids])
        chosen = np.lexsort((ids,dist))[:k]
        ids, dist, frac = ids[chosen], dist[chosen], frac[chosen]
        valid = bool(gap[i]>0 and movement[i]>=1)
        diff = np.abs((np.rad2deg(np.arctan2(v[ids,1],v[ids,0])-heading[i])+180)%360-180) if valid else np.zeros(len(ids))
        rows.extend(zip([trip.trip_id]*len(ids),[i]*len(ids),ids,dist,diff,[speed[i]]*len(ids),lengths[ids],[max(gap[i],0)]*len(ids),frac,[int(valid)]*len(ids),[int(gap[i]>0)]*len(ids)))
    return pd.DataFrame(rows,columns=['trip_id','observation_id','arc_id',*FEATURE_COLUMNS])
