import argparse
from pathlib import Path
import folium
from preprocessing.load_data import load_trip, load_reference

def create_map(directory, output, predicted=None):
    trip=load_trip(directory)
    gps=trip.track[:,:2][:,::-1]
    nodes=trip.nodes[:,::-1]
    m=folium.Map(location=gps.mean(axis=0).tolist(),zoom_start=13)
    roads=folium.FeatureGroup(name='Road network',show=False)
    # One multi-polyline reduces HTML overhead compared with one layer per arc.
    folium.PolyLine(nodes[trip.arcs].tolist(),color='#888888',weight=1,opacity=.4).add_to(roads); roads.add_to(m)
    obs=folium.FeatureGroup(name='GPS observations'); folium.PolyLine(gps.tolist(),color='#2563eb',weight=2).add_to(obs)
    for coord in gps: folium.CircleMarker(coord.tolist(),radius=2,color='#2563eb',fill=True).add_to(obs)
    obs.add_to(m)
    ref=Path(directory)/f'{trip.trip_id}.route'
    if ref.exists():
        route=load_reference(directory,len(trip.arcs)); group=folium.FeatureGroup(name='Reference route'); folium.PolyLine(nodes[trip.arcs[route]].tolist(),color='#16803c',weight=4).add_to(group); group.add_to(m)
    if predicted is not None:
        group=folium.FeatureGroup(name='Predicted route'); folium.PolyLine(nodes[trip.arcs[predicted]].tolist(),color='#e85d04',weight=4).add_to(group); group.add_to(m)
    folium.LayerControl().add_to(m); m.fit_bounds(gps.tolist()); m.save(str(output))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('output'); args=p.parse_args(); create_map(args.directory,args.output)
