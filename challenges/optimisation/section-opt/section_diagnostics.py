"""Plots for correction passes; all tracking here is diagnostic, not optimizer data."""
import numpy as np
import torch
import cheetah
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from .section_measurement import section_readout
from .section_response import track_to_boundary


def plot_section_convergence(record, session):
    if session is None:
        print(record['output_screen'], record['status'])
        return
    values = np.array([r['value'] for r in session.records])
    selected = record['selected_evaluation']
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(1,len(values)+1),values,alpha=.45,label='measured objective')
    ax.plot(np.arange(1,len(values)+1),np.minimum.accumulate(values),label='best measured')
    ax.scatter([selected],[values[selected-1]],c='red',zorder=5,label='restored setting')
    ax.set(title=record['input_screen']+' -> '+record['output_screen'],
           xlabel='Screen image number',ylabel='Local objective',yscale='symlog')
    ax.legend();fig.tight_layout();plt.show();plt.close(fig)


def envelope(lattice, beam):
    current = beam.clone()
    distance = 0.
    rows, apertures = [], []
    def record(name):
        moments, survival = section_readout(current)
        rows.append([distance,*moments,survival])
    record('input')
    with torch.no_grad():
        for element in lattice.elements:
            current = element.track(current)
            distance += float(element.length)
            if isinstance(element,cheetah.Aperture) and element.is_active:
                apertures.append([distance,float(element.x_max),float(element.y_max)])
            record(element.name)
    return np.asarray(rows),np.asarray(apertures).reshape(-1,3)


def plot_pass_envelopes(run, reference, actual_root, ideal_root, *, title='Correction pass'):
    before, _ = envelope(run['initial_lattice'],actual_root)
    after, limits = envelope(run['lattice'],actual_root)
    design, _ = envelope(reference,ideal_root)
    fig, axes = plt.subplots(2,2,figsize=(15,9),sharex=True)
    for plane,(mu,size,label) in enumerate(((1,3,'x'),(2,4,'y'))):
        ax = axes[0,plane]
        for data,name,color in ((before,'initial','tab:orange'),(after,'corrected','tab:blue'),(design,'design','tab:green')):
            ax.plot(data[:,0],data[:,mu]*1e3,label=name+' orbit',color=color)
            ax.fill_between(data[:,0],(data[:,mu]-2*data[:,size])*1e3,
                            (data[:,mu]+2*data[:,size])*1e3,color=color,alpha=.12,label=name+' ±2σ')
            axes[1,plane].plot(data[:,0],data[:,size]*1e3,label=name,color=color)
        if len(limits):
            # Actual aperture checks in the loaded lattice, including narrow collimators.
            for sign in (-1,1):
                ax.step(limits[:,0],sign*limits[:,plane+1]*1e3,where='post',color='black',ls='--',
                        label='aperture limits' if sign==1 else None)
            axes[1,plane].step(limits[:,0],limits[:,plane+1]*1e3,where='post',color='black',ls='--',label='half aperture')
        ax.set(ylabel=f'{label} orbit / envelope [mm]',yscale='symlog')
        axes[1,plane].set(xlabel='s from input [m]',ylabel=f'σ{label} [mm]',yscale='log')
        ax.legend(fontsize=8);axes[1,plane].legend(fontsize=8)
    fig.suptitle(title+' — aperture checks shown as dashed limits');fig.tight_layout();plt.show();plt.close(fig)
    return dict(initial=before,final=after,design=design,apertures=limits)


def plot_exit_beams(run, reference, actual_root, ideal_root, screen):
    outputs=[track_to_boundary(run['initial_lattice'],actual_root,screen),
             track_to_boundary(run['lattice'],actual_root,screen),
             track_to_boundary(reference,ideal_root,screen)]
    target,_=section_readout(outputs[-1])
    if not np.isfinite(target).all() or np.any(target[2:]<=0):
        print(f'{screen}: no valid design image; cannot normalize the exit plots.')
        return
    clouds=[]
    for beam in outputs:
        live=beam.survival_probabilities.detach().cpu().numpy()>0
        xy=beam.particles.detach().cpu().numpy()[live][:,[0,2]]
        clouds.append(xy)
    nonempty=[xy for xy in clouds if len(xy)]
    anchors=np.array([target[:2]-3*target[2:],target[:2]+3*target[2:]])
    allxy=np.concatenate(nonempty+[anchors])
    extent=[(allxy[:,j].min()*1e3,allxy[:,j].max()*1e3) for j in range(2)]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    norm=np.concatenate([(xy-target[:2])/target[2:] for xy in nonempty]+[np.array([[-3.,-3.],[3.,3.]])])
    radius=max(3.,float(np.abs(norm).max()))
    for j,(xy,name,beam) in enumerate(zip(clouds,('initial','corrected','design'),outputs)):
        axes[0,j].hist2d(xy[:,0]*1e3,xy[:,1]*1e3,bins=60,range=extent)
        axes[0,j].add_patch(Ellipse(target[:2]*1e3,4*target[2]*1e3,4*target[3]*1e3,
                                    fill=False,color='red'))
        z=(xy-target[:2])/target[2:]
        axes[1,j].hist2d(z[:,0],z[:,1],bins=60,range=[(-radius,radius)]*2)
        axes[1,j].add_patch(Circle((0,0),2,fill=False,color='red'))
        axes[1,j].set_aspect('equal')
        axes[0,j].set(title=f'{name}: survival {float(beam.survival_probabilities.mean()):.1%}',xlabel='x [mm]',ylabel='y [mm]')
        axes[1,j].set(xlabel='(x − design μx) / design σx',ylabel='(y − design μy) / design σy')
    fig.suptitle(screen+' — red: design 2σ ellipse; circle in normalized coordinates')
    fig.tight_layout();plt.show();plt.close(fig)
