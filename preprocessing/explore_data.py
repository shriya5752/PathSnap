import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .load_data import load_trip, load_reference
from .create_features import project, candidate_features
from .create_labels import reference_alignment, attach_labels

def summarize(trip, route, directory, split):
    t,n,a=trip.track,trip.nodes,trip.arcs
    gaps=np.diff(t[:,2]); p,xy=project(trip)
    graph=coo_matrix((np.ones(len(a)),(a[:,0],a[:,1])),shape=(len(n),len(n))).tocsr()
    nc, labels=connected_components(graph,directed=False)
    sizes=np.bincount(labels)
    return dict(split=split,trip_id=trip.trip_id,gps_points=len(t),nodes=len(n),arcs=len(a),reference_arcs=len(route),files=4,total_bytes=sum(f.stat().st_size for f in directory.glob('*')),missing_values=0,invalid_coordinates=0,invalid_ids=0,nonpositive_gaps=int((gaps<=0).sum()),one_second_gaps=int((gaps==1).sum()),gaps_gt_5s=int((gaps>5).sum()),max_gap_s=float(gaps.max()) if len(gaps) else 0,median_gap_s=float(np.median(gaps)) if len(gaps) else 0,lon_min=t[:,0].min(),lon_max=t[:,0].max(),lat_min=t[:,1].min(),lat_max=t[:,1].max(),observed_path_m=np.linalg.norm(np.diff(p,axis=0),axis=1).sum(),weak_components=nc,isolated_nodes=int((np.bincount(a.ravel(),minlength=len(n))==0).sum()),largest_component_nodes=int(sizes.max()),reference_discontinuities=int((a[route[:-1],1]!=a[route[1:],0]).sum()),zero_length_arcs=int((np.linalg.norm(xy[a[:,1]]-xy[a[:,0]],axis=1)==0).sum()))

def trip_plot(trip, route, output):
    p,n=project(trip)
    fig,ax=plt.subplots(figsize=(8,7))
    ax.add_collection(LineCollection(n[trip.arcs],colors='#b9c3ca',linewidths=.35,label='Road network'))
    ax.add_collection(LineCollection(n[trip.arcs[route]],colors='#198754',linewidths=1.7,label='Reference route'))
    ax.scatter(p[:,0],p[:,1],s=5,c='#275dad',alpha=.7,label='GPS observations')
    ax.autoscale(); ax.set_aspect('equal'); ax.set(xlabel='East (m)',ylabel='North (m)',title=f'Trip {trip.trip_id}: GPS and road network'); ax.legend(); fig.tight_layout(); fig.savefig(output,dpi=150); plt.close(fig)

def plots(summary, output):
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for split,group in summary.groupby('split'):
        axs[0,0].hist(group.gps_points,bins=20,alpha=.65,label=split)
        axs[0,1].scatter(group.nodes,group.arcs,s=16,label=split)
        axs[1,0].scatter(group.gps_points,group.max_gap_s,s=16,label=split)
        axs[1,1].hist(group.weak_components,bins=20,alpha=.65,label=split)
    for ax,title,x,y in zip(axs.ravel(),['GPS observations per trip','Road-network size','Timestamp anomalies','Weak components per trip'],['GPS points','Nodes','GPS points','Components'],['Trips','Arcs','Maximum gap (s)','Trips']):
        ax.set(title=title,xlabel=x,ylabel=y); ax.legend()
    fig.tight_layout(); fig.savefig(output/'dataset_overview.png',dpi=150); plt.close(fig)

def run(root, output, k=10, shortlist=64):
    root,output=Path(root),Path(output); output.mkdir(parents=True,exist_ok=True)
    summaries=[]; quality=[]
    for split in ['train','student_test']:
        folders=sorted(p for p in (root/split).iterdir() if p.is_dir())
        if len(folders)!= {'train':80,'student_test':10}[split]: raise ValueError(f'Unexpected trip count: {split}')
        for i,d in enumerate(folders):
            trip=load_trip(d); route=load_reference(d,len(trip.arcs))
            summaries.append(summarize(trip,route,d,split))
            f=candidate_features(trip,k,shortlist); alignment=reference_alignment(trip,route)
            labeled=attach_labels(f,alignment)
            hit=labeled.groupby('observation_id').label.max().reindex(alignment.observation_id)
            usable=alignment.label_status.eq('usable')
            quality.append(dict(split=split,trip_id=d.name,observations=len(alignment),usable=int(usable.sum()),ambiguous=int(alignment.label_status.eq('ambiguous').sum()),far=int(alignment.label_status.eq('far').sum()),candidate_misses=int((hit==0).sum()),candidate_rows=len(f)))
            f.to_csv(output/f'{split}_features.csv',mode='w' if i==0 else 'a',header=i==0,index=False,float_format='%.5f')
            labeled.to_csv(output/f'{split}_labeled.csv',mode='w' if i==0 else 'a',header=i==0,index=False,float_format='%.5f')
            alignment.to_csv(output/f'{split}_alignment.csv',mode='w' if i==0 else 'a',header=i==0,index=False,float_format='%.5f')
            if i==0: trip_plot(trip,route,output/f'{split}_trajectory.png')
            print(split,d.name,len(f),flush=True)
    summary=pd.DataFrame(summaries); summary.to_csv(output/'dataset_summary.csv',index=False)
    quality_df=pd.DataFrame(quality)
    quality_df.to_csv(output/'label_quality.csv',index=False)
    cols=['gps_points','nodes','arcs','reference_arcs','missing_values','nonpositive_gaps','one_second_gaps','gaps_gt_5s','isolated_nodes','reference_discontinuities']
    grouped=summary.groupby('split')
    aggregate=grouped[cols].sum()
    aggregate.insert(0,'trips',grouped.size())
    aggregate['max_gap_s']=grouped.max_gap_s.max()
    aggregate['max_weak_components']=grouped.weak_components.max()
    aggregate.to_csv(output/'partition_summary.csv')
    quality_df.groupby('split')[['observations','usable','ambiguous','far','candidate_misses','candidate_rows']].sum().to_csv(output/'label_summary.csv')
    plots(summary,output)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',default='results/person1'); p.add_argument('--k',type=int,default=10); p.add_argument('--shortlist',type=int,default=64); args=p.parse_args(); run(args.data,args.output,args.k,args.shortlist)
