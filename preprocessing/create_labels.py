import numpy as np
import pandas as pd
from .create_features import project, segment_distance

def reference_alignment(trip, route, max_distance=50.0, ambiguity_margin=2.0):
    """Weak labels: monotone minimum squared-distance alignment to route occurrences.

    Repeated arc IDs remain distinct occurrences during DP. No reference used in
    candidate generation. Close competing arcs are flagged, not asserted as truth.
    """
    p,n = project(trip)
    a = n[trip.arcs[route,0]]
    v = n[trip.arcs[route,1]]-a
    m = len(route)
    back = np.empty((len(p),m), dtype=np.int32)
    cost = np.zeros(m)
    idx = np.arange(m)
    for t,point in enumerate(p):
        dist,_ = segment_distance(point,a,v)
        prefix = np.minimum.accumulate(cost)
        predecessor = np.maximum.accumulate(np.where(cost==prefix,idx,0))
        back[t] = predecessor
        cost = prefix + dist**2
    states = np.empty(len(p),dtype=np.int32)
    states[-1] = np.argmin(cost)
    for t in range(len(p)-1,0,-1):
        states[t-1] = back[t,states[t]]
    rows=[]
    for t,(point,j) in enumerate(zip(p,states)):
        dist,_ = segment_distance(point,a,v)
        other = dist[route != route[j]]
        margin = (other.min()-dist[j]) if len(other) else np.inf
        status = 'far' if dist[j]>max_distance else ('ambiguous' if margin<ambiguity_margin else 'usable')
        rows.append((trip.trip_id,t,int(route[j]),int(j),float(dist[j]),float(margin),status))
    return pd.DataFrame(rows,columns=['trip_id','observation_id','reference_arc_id','route_position','reference_distance_m','margin_m','label_status'])

def attach_labels(features, alignment):
    keys=['trip_id','observation_id']
    out=features.merge(alignment[keys+['reference_arc_id','label_status']],on=keys,validate='many_to_one')
    out['label']=(out.arc_id==out.reference_arc_id).astype('Int8')
    out.loc[out.label_status!='usable','label']=pd.NA
    return out
