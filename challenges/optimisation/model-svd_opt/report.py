"""Render measured progress, using saved public observations only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(directory):
    directory=Path(directory);rows=[json.loads(x) for x in (directory/'shots.jsonl').read_text().splitlines()]
    summary=json.loads((directory/'summary.json').read_text());target=np.array(summary['config']['target'])*1e3
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for method,color in [('bo','tab:blue'),('rcds','tab:orange')]:
        r=[r for r in rows if r['branch']==method];x=np.array([r['info']['evaluations'] for r in r]);valid=np.array([r['info']['on_screen'] for r in r])
        def values(key):return np.array([a['info'][key] for a in r])
        axes[0,0].plot(x,values('transmission'),label=method.upper(),color=color)
        axes[0,1].plot(x,np.where(valid,np.hypot(values('mu_x'),values('mu_y'))*1e3,np.nan),label=method.upper(),color=color)
        for p,style in [('x','-'),('y','--')]:axes[1,0].plot(x,np.where(valid,values('sigma_'+p)*1e3,np.nan),style,color=color,alpha=.8,label=method.upper()+' '+p)
        from_prefix=[]
        for a in r:
            mask=np.asarray(a['obs']['bpm_valid'],bool);bad=np.flatnonzero(~mask);from_prefix.append(int(bad[0]) if len(bad) else len(mask))
        axes[1,1].step(x,from_prefix,where='post',label=method.upper(),color=color)
    axes[1,0].axhline(target[0],color='k',alpha=.6,ls=':',label='target x')
    if target[1]!=target[0]:axes[1,0].axhline(target[1],color='gray',ls=':',label='target y')
    for ax,title,ylabel in zip(axes.ravel(),['Transmission','Screen centering','Measured RMS sizes','Contiguous valid BPMs'],['fraction','centroid radius [mm]','RMS [mm]','BPM count']):
        ax.set(title=title,xlabel='SimEnv steps per branch (replay included)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f"{summary['config']['backend']}, seed {summary['config']['seed']}, target ({target[0]:g}, {target[1]:g}) mm")
    fig.savefig(directory/'progress.png',dpi=160);plt.close(fig)
    return directory/'progress.png'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');plot(p.parse_args().directory)
