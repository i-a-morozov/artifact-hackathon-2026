"""Safe affine singular-mode controls: no componentwise magnet clipping."""
import numpy as np

class ModalSpace:
    def __init__(self,names,base,basis,limits,caps):
        self.names=list(names);self.base=np.array([base[n] for n in names]);self.basis=np.asarray(basis,float)
        bounds=np.array([limits[n] for n in names]);self.bounds=bounds
        if np.any(self.base<bounds[:,0]-1e-9) or np.any(self.base>bounds[:,1]+1e-9):raise ValueError('Base outside machine limits')
        count=np.maximum(1,(np.abs(self.basis)>1e-14).sum(axis=1))
        low=[];high=[]
        for j,col in enumerate(self.basis.T):
            active=np.abs(col)>1e-14
            a=(bounds[active,0]-self.base[active])/count[active]/col[active]
            b=(bounds[active,1]-self.base[active])/count[active]/col[active]
            low.append(max(-caps[j],np.max(np.minimum(a,b),initial=-np.inf)))
            high.append(min(caps[j],np.min(np.maximum(a,b),initial=np.inf)))
        low=np.array(low);high=np.array(high);self.active=np.flatnonzero(high-low>1e-12)
        self.low=low[self.active];self.high=high[self.active];self.basis=self.basis[:,self.active]
        self.start=-self.low/(self.high-self.low)
    def settings(self,u):
        u=np.asarray(u,float)
        if u.shape!=self.start.shape or not np.isfinite(u).all() or np.any(u< -1e-8) or np.any(u>1+1e-8):raise ValueError('Invalid modal coordinate')
        value=self.base+self.basis@(self.low+np.clip(u,0,1)*(self.high-self.low))
        if np.any(value<self.bounds[:,0]-1e-8) or np.any(value>self.bounds[:,1]+1e-8):raise ValueError('Modal setting violates native limit')
        return dict(zip(self.names,map(float,value)))
    def candidate(self,response,residual):
        physical=np.linalg.lstsq(response@self.basis,-np.asarray(residual),rcond=1e-7)[0]
        # Scale the complete direction to the box boundary; preserve its direction.
        delta=physical/(self.high-self.low);scale=1.
        for u,d in zip(self.start,delta):
            if d>0:scale=min(scale,(1-u)/d)
            elif d<0:scale=min(scale,-u/d)
        return np.clip(self.start+max(0.,scale)*delta,0,1)
    def metadata(self):
        return dict(names=self.names,base=self.base,basis=self.basis,low=self.low,high=self.high,active_modes=self.active)

def prefix(valid):
    missing=np.flatnonzero(~np.asarray(valid,dtype=bool))
    return int(missing[0]) if len(missing) else len(valid)

def objective(obs,info,stage,target,count=None):
    if stage=='thread':
        valid=np.asarray(obs['bpm_valid'][:count],bool)
        pos=np.r_[np.asarray(obs['bpm_x'][:count])[valid],np.asarray(obs['bpm_y'][:count])[valid]]
        # Missing values are NEVER interpreted as zero-position measurements.
        error=float(np.mean((pos/.001)**2)) if pos.size else 1e6
        ip_penalty=100.*(not info['on_screen']) if count==len(obs['bpm_valid']) else 0.
        return 100.*np.count_nonzero(~valid)+20.*error/(20.+error)+ip_penalty+2*(1-info['transmission'])
    if not info['on_screen'] or min(info['sigma_x'],info['sigma_y'])<=0:return 1000.+100*(1-info['transmission'])
    if stage=='optics':return float(np.sum(np.log(np.array([info['sigma_x'],info['sigma_y']])/target)**2)+2*(1-info['on_screen_fraction']))
    error=float(np.sum((np.array([info['mu_x'],info['mu_y']])/target)**2))
    return 900.*error/(900.+error)+2*(1-info['on_screen_fraction'])
