"""Ideal calibrations only; never inspect the stochastic episode's hidden state."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import cheetah
from optics_matching_vbc_on import MatchedOptics,build_cheetah,build_xsuite
from nominal_model import xsuite_model as nominal
from orbit_matching.vbc_reference import ReferenceOrbitEngine

class Models:
    def __init__(self,backend,target,directory):
        self.optics=MatchedOptics(l4h_off=True);self.target=target;self.backend=backend
        self.engine=ReferenceOrbitEngine(backend,self.optics)
        self.orbit=self.engine.evaluate(*target)
        self.directory=Path(directory)
        self.names=self.optics.quadrupoles
    def orbit_at(self,settings):
        original=self.engine.optics
        class Fixed:
            def evaluate(_,x,y):return dict(settings={n:settings[n] for n in self.names})
        try:self.engine.optics=Fixed();return self.engine.evaluate(*self.target)
        finally:self.engine.optics=original
    def beta_response(self):
        # A separate linear optics calculation, not stochastic bunch moments.
        # Symmetric trajectory probes transport the nominal projected Twiss ellipse.
        segment=build_cheetah(*self.target,model=self.optics,dtype=torch.float64)
        elements={e.name:e for e in segment.elements}
        screens=[e.name for e in segment.elements if '_dia_scr_' in e.name or '_dcp_scr_' in e.name]
        if nominal.READOUT not in screens:screens.append(nominal.READOUT)
        chunks=[];start=0
        for i,e in enumerate(segment.elements):
            if e.name in screens:chunks.append(cheetah.Segment(segment.elements[start:i+1]));start=i+1
        coords,energy,_=nominal.load_distribution();twiss=nominal.twiss_from_coordinates(coords)
        test=torch.zeros((9,7),dtype=torch.float64);test[:,-1]=1;h=1e-7
        for j in range(4):test[1+2*j,j]=h;test[2+2*j,j]=-h
        def beta():
            beam=cheetah.ParticleBeam(particles=test.clone(),energy=torch.tensor(energy,dtype=torch.float64));out=[]
            with torch.no_grad():
                for chunk in chunks:
                    beam=chunk.track(beam);r=((beam.particles[1:9:2,:4]-beam.particles[2:9:2,:4])/(2*h)).numpy().T
                    values=[]
                    for p,j in [('x',0),('y',2)]:
                        a,b,c,d=r[j:j+2,j:j+2].ravel();b0=twiss[f'beta_{p}'];a0=twiss[f'alpha_{p}']
                        values.append((a*a*b0-2*a*b*a0+b*b*(1+a0*a0)/b0)/(a*d-b*c))
                    out.extend(values)
            return np.log(out)
        if self.backend=='xsuite':
            converted=build_xsuite(*self.target,model=self.optics,apertures=False)
            elements={n:converted.line[n] for n in self.names}
            def beta():
                optics=nominal.optics_along(converted,twiss)
                slots=[optics['element'].tolist().index(n) for n in screens]
                return np.log(np.column_stack([optics['beta_x'][slots],optics['beta_y'][slots]]).ravel())
        R=np.empty((2*len(screens),len(self.names)));check=np.empty_like(R);step=1e-4
        for j,n in enumerate(self.names):
            e=elements[n];k=e.k1.clone() if self.backend=='cheetah' else float(e.k1);der=[]
            def write(value):
                if self.backend=='cheetah':e.k1.copy_(value)
                else:e.k1=float(value)
            for dh in (step,step/2):
                write(k+dh);plus=beta();write(k-dh);minus=beta();der.append((plus-minus)/(2*dh))
            write(k);check[:,j]=der[0];R[:,j]=(4*der[1]-der[0])/3
        w=np.full(len(screens)*2,.5/(2*(len(screens)-1)))
        ip=screens.index(nominal.READOUT);w[2*ip:2*ip+2]=.25
        U,s,Vt=np.linalg.svd(np.sqrt(w)[:,None]*R,full_matrices=False)
        rank=int(np.count_nonzero(s>s[0]*1e-8))
        result=dict(response=R,row_weights=w,singular_values=s,Vt=Vt,U=U,rank=rank,quadrupoles=np.array(self.names),screens=np.array(screens),step=step,
                    relative_step_change=np.linalg.norm(R-check)/np.linalg.norm(R),backend=np.array(self.backend+'_float64'),optics_fingerprint=np.array(self.optics.configuration_fingerprint()))
        np.savez_compressed(self.directory/'optics_response.npz',**result)
        return result
    def save_orbits(self,data,label):
        for p in 'xy':
            r=data[p];U,s,Vt=np.linalg.svd(r['weighted_response'],full_matrices=False)
            np.savez_compressed(self.directory/f'{label}_{p}.npz',**r,U=U,singular_values=s,Vt=Vt)
