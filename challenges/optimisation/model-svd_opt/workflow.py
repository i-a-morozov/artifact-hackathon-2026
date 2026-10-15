"""Stochastic SimEnv workflow with paired BO/RCDS optics branches."""
from pathlib import Path
from dataclasses import dataclass,asdict
import importlib.util
import sys
import json
import time
import numpy as np
from .controls import ModalSpace,prefix,objective
from .models import Models

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'RL_framework'))
from rl_framework.config import RLConfig
from rl_framework.envs.sim_env import SimEnv
from optics_matching_vbc_on import MatchedOptics,lattice_document

spec=importlib.util.spec_from_file_location('section_opt',ROOT/'section-opt/__init__.py',submodule_search_locations=[str(ROOT/'section-opt')])
section=importlib.util.module_from_spec(spec);sys.modules['section_opt']=section;spec.loader.exec_module(section)
from section_opt.beam_tuning import rcds_search,BudgetExhausted
from section_opt.section_measurement import full_bo_search

@dataclass
class Config:
    seed:int=42
    target:tuple=(.001,.001)
    backend:str='cheetah'
    particles:int=10000
    repeats:int=3
    thread_budget:int=40
    optics_budget:int=35
    center_budget:int=25
    beam_variation:float=1.
    misalignment_variation:float=1.
    beam_jitter:float=.1
    ripple:float=.001
    bpm_noise:float=0.
    steering_cap:float=.002
    quad_cap:float=2.

def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value

class Machine:
    def __init__(self,env,out,repeats):
        self.env=env;self.out=out;self.repeats=repeats;self.records=[];self.actions=[];self.branch='thread';self.truncated=False
    def record(self,obs,info,stage,action=None,wall_time_s=None):
        row=dict(branch=self.branch,stage=stage,shot=len(self.records),wall_time_s=wall_time_s,action=action,obs={k:obs[k] for k in ('bpm_x','bpm_y','bpm_valid')},info=info)
        self.records.append(clean(row))
        with (self.out/'shots.jsonl').open('a') as f:f.write(json.dumps(clean(row),allow_nan=False)+'\n')
    def reset(self,seed):
        self.truncated=False;started=time.perf_counter();obs,info=self.env.reset(seed=seed);duration=time.perf_counter()-started;self.last=(obs,info);self.record(obs,info,'reset',wall_time_s=duration);return self.last
    def shot(self,settings,stage):
        if self.truncated:raise BudgetExhausted('SimEnv episode truncated')
        action=dict(magnets=settings,screen=self.env.config.target_screen)
        started=time.perf_counter()
        obs,_,terminated,self.truncated,info=self.env.step(action)
        duration=time.perf_counter()-started
        if terminated:self.truncated=True
        self.actions.append(clean(action));self.last=(obs,info);self.record(obs,info,stage,action,wall_time_s=duration)
        # SimEnv clips native controls. Treat any unexpected clipping as an error.
        for n,v in settings.items():
            if not np.isclose(info['magnets'][n],v,rtol=2e-6,atol=1e-9):raise RuntimeError(f'Unexpected control clipping: {n}')
        return obs,info
    def sample(self,settings,stage,count=None):
        samples=[self.shot(settings,stage) for _ in range(self.repeats)]
        valid=np.array([o['bpm_valid'] for o,_ in samples],bool);majority=valid.sum(axis=0)>self.repeats/2
        obs=dict(bpm_valid=majority)
        for p in 'xy':
            values=np.array([o[f'bpm_{p}'] for o,_ in samples]);obs[f'bpm_{p}']=np.divide((values*valid).sum(axis=0),valid.sum(axis=0),out=np.zeros(values.shape[1]),where=valid.sum(axis=0)>0)
        info=dict(samples[-1][1]);info['on_screen']=sum(i['on_screen'] for _,i in samples)>self.repeats/2
        for key in ('transmission','on_screen_fraction','mu_x','mu_y','sigma_x','sigma_y'):
            values=[i[key] for _,i in samples if i['on_screen']] if key in ('mu_x','mu_y','sigma_x','sigma_y') else [i[key] for _,i in samples]
            info[key]=float(np.mean(values)) if values else 0.
        for key in ('benchmark_score','scored_mismatch','scored_relative_error'):
            scores=np.array([i[key] for _,i in samples])
            info[key+'_mean']=float(scores.mean())
            info[key+'_standard_error']=float(scores.std(ddof=1)/np.sqrt(len(scores))) if len(scores)>1 else 0.
        values=np.array([objective(o,i,stage,np.asarray(self.env.config.target[2:]),count) for o,i in samples])
        return float(values.mean()),float(max(values.std(ddof=1)/np.sqrt(len(values)),1e-8)) if len(values)>1 else 1e-8,obs,info

class Session:
    def __init__(self,machine,space,stage,budget,*,count=None,stop=None):
        self.machine=machine;self.space=space;self.stage=stage;self.budget=budget;self.count=count;self.stop=stop;self.records=[];self.stopped=False
    @property
    def remaining(self):return 0 if self.stopped else self.budget-len(self.records)
    def measure(self,u):
        if self.remaining<=0:raise BudgetExhausted()
        value,error,obs,info=self.machine.sample(self.space.settings(u),self.stage,self.count)
        self.records.append(dict(u=np.array(u),value=value,error=error,obs=obs,info=info))
        if self.stop and self.stop(obs,info):self.stopped=True
        return value,error
    def best(self):return min(self.records,key=lambda r:r['value'])
    def search(self,method,seed):
        if not len(self.space.start):
            self.measure(self.space.start);return self.best()
        if method=='bo':full_bo_search(self,self.space.start,seed=seed)
        else:rcds_search(self.measure,self.space.start,self.remaining,seed=seed)
        return self.best()
    def save(self,path):path.write_text(json.dumps(clean(dict(space=self.space.metadata(),records=self.records)),indent=2,allow_nan=False)+'\n')

def steering_space(machine,models,count,config):
    names=[];blocks=[];caps=[];responses=[]
    for p in 'xy':
        r=models.orbit[p];R=r['response'][:count]
        svd_matrix=r['weighted_response'] if count==len(machine.env.bpm_names) else R
        _,s,Vt=np.linalg.svd(svd_matrix,full_matrices=False)
        n=min(3,np.count_nonzero(s>s[0]*1e-8)) if len(s) and s[0]>0 else 0
        names.extend([name+('_h' if p=='x' else '_v') for name in r['correctors']]);blocks.append(Vt[:n].T)
        caps.extend(np.minimum(config.steering_cap,.01/np.maximum(s[:n],1e-12)))
        responses.append(R)
    from scipy.linalg import block_diag
    basis=block_diag(*blocks);R=block_diag(*responses)
    return ModalSpace(names,machine.env.magnet_settings(),basis,machine.env.magnet_limits,caps),R

def thread(machine,models,config,out):
    # Fixed objective within each prefix. New prefix => new SVD and fresh search.
    used=0;iteration=0;obs,info=machine.last;total=len(obs['bpm_valid']);history=[]
    while used+1<config.thread_budget and not machine.truncated:
        before=prefix(obs['bpm_valid']);count=min(total,before+1)
        space,R=steering_space(machine,models,count,config)
        stop=lambda o,i: prefix(o['bpm_valid'])>=count and (count<total or i['on_screen'])
        budget=min(config.thread_budget-used-1,max(5,2*len(space.start)+3))
        session=Session(machine,space,'thread',budget,count=count,stop=stop)
        session.measure(space.start)
        if session.remaining and len(space.start):
            row=session.records[-1];valid=row['obs']['bpm_valid'][:count]
            mask=np.r_[valid,valid];res=np.r_[row['obs']['bpm_x'][:count],row['obs']['bpm_y'][:count]]
            if mask.any():session.measure(space.candidate(R[mask],res[mask]))
        if session.remaining:
            if len(space.start):rcds_search(session.measure,space.start,session.remaining,seed=config.seed+iteration)
            else:session.measure(space.start)
        # Prefer a confirmed advancing shot before objective minima at an old prefix.
        advancing=[r for r in session.records if stop(r['obs'],r['info'])]
        best=min(advancing or session.records,key=lambda r:r['value'])
        _,_,obs,info=machine.sample(space.settings(best['u']),'thread',count)
        used+=len(session.records)+1;iteration+=1
        session.save(out/f'threading_{iteration:02d}.json')
        history.append(dict(prefix_before=before,objective_prefix=count,prefix_after=prefix(obs['bpm_valid']),on_screen=info['on_screen']))
        print(f"Thread {iteration}: BPM prefix {before}->{prefix(obs['bpm_valid'])}/{total}, IP={info['on_screen']}",flush=True)
        if info['on_screen'] and prefix(obs['bpm_valid'])==total:break
    return dict(success=bool(info['on_screen'] and prefix(obs['bpm_valid'])==total),evaluations=used,progress=history,final=info)

def optics_stage(machine,models,beta,config,method,out):
    n=min(5,int(beta['rank']));s=beta['singular_values'][:n]
    space=ModalSpace(models.names,machine.env.magnet_settings(),beta['Vt'][:n].T,machine.env.magnet_limits,np.minimum(config.quad_cap,2./np.maximum(s,1e-12)))
    session=Session(machine,space,'optics',config.optics_budget)
    best=session.search(method,config.seed)
    _,_,obs,info=machine.sample(space.settings(best['u']),'optics')
    session.save(out/f'{method}_optics.json')
    return info

def center_stage(machine,models,config,method,out):
    settings=machine.env.magnet_settings();orbit=models.orbit_at(settings);models.save_orbits(orbit,f'{method}_final_optics_orbit')
    names=[];R=np.zeros((2,4))
    for i,p in enumerate('xy'):
        indices=[len(orbit[p]['correctors'])-2,len(orbit[p]['correctors'])-1]
        names.extend([orbit[p]['correctors'][j]+('_h' if p=='x' else '_v') for j in indices]);R[i,2*i:2*i+2]=orbit[p]['response'][-1,indices]
    space=ModalSpace(names,settings,np.eye(4),machine.env.magnet_limits,np.full(4,np.inf))
    session=Session(machine,space,'center',config.center_budget)
    session.measure(space.start)
    if session.remaining and session.records[-1]['info']['on_screen']:
        info=session.records[-1]['info'];session.measure(space.candidate(R,np.array([info['mu_x'],info['mu_y']])))
    if session.remaining:rcds_search(session.measure,space.start,session.remaining,seed=config.seed)
    best=session.best();_,_,obs,info=machine.sample(space.settings(best['u']),'center')
    session.save(out/f'{method}_centering.json')
    return info

def run(config,out):
    run_started=time.perf_counter();timings={}
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    if min(config.thread_budget,config.optics_budget,config.center_budget,config.repeats,config.particles)<1:raise ValueError('Budgets, particles and repeats must be positive')
    provider=MatchedOptics(l4h_off=True);provider.evaluate(*config.target)
    lattice=out/'episode_lattice.json';lattice.write_text(json.dumps(lattice_document(*config.target,model=provider),indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(clean(asdict(config)),indent=2)+'\n')
    print('Building nominal response controls',flush=True)
    models=Models(config.backend,config.target,out);models.save_orbits(models.orbit,'nominal_orbit');beta=models.beta_response()
    timings['nominal_setup_and_calibration_s']=time.perf_counter()-run_started
    maxsteps=config.repeats*(config.thread_budget+config.optics_budget+config.center_budget+20)
    cfg=RLConfig(backend=config.backend,nominal_lattice_json=str(lattice),num_particles=config.particles,target=(0.,0.,*config.target),max_episode_steps=maxsteps,
        beam_variation=config.beam_variation,misalignment_variation=config.misalignment_variation,beam_jitter=config.beam_jitter,power_supply_ripple=config.ripple,bpm_noise=config.bpm_noise,
        include_cavity_phases=False,include_vbc_chicane=False)
    stage_started=time.perf_counter()
    env=SimEnv(cfg);machine=Machine(env,out,config.repeats);machine.reset(config.seed)
    # Response and observations must refer to the same named diagnostics.
    if env.bpm_names!=models.orbit['x']['monitors'][:-1]:raise RuntimeError('BPM order mismatch')
    timings['initial_environment_and_reset_s']=time.perf_counter()-stage_started
    stage_started=time.perf_counter()
    threaded=thread(machine,models,config,out);timings['threading_s']=time.perf_counter()-stage_started;actions=list(machine.actions);reference=list(machine.records)
    summary=dict(config=asdict(config),optics_fingerprint=provider.configuration_fingerprint(),threading=threaded,branches={},pairing=dict(thread_actions=len(actions)))
    if threaded['success']:
        for method in ('bo','rcds'):
            branch_started=time.perf_counter();branch_timings={}
            machine.branch=method
            # Reconstruct both branches, so transport caches have the same lifecycle.
            machine.env.close()
            machine.env=SimEnv(cfg)
            machine.reset(config.seed)
            replay_errors=[]
            for j,action in enumerate(actions):
                o,i=machine.shot(action['magnets'],'thread_replay');expected=reference[j+1]
                if not np.array_equal(o['bpm_valid'],expected['obs']['bpm_valid']):raise RuntimeError('Seeded threading replay changed BPM availability')
                delta=max(float(np.max(np.abs(np.asarray(o[key])-expected['obs'][key]))) for key in ('bpm_x','bpm_y'))
                replay_errors.append(delta)
                # Record numerical replay differences explicitly; never claim
                # bitwise identity when it has not been observed.
                if delta>1e-4:raise RuntimeError('Seeded threading replay differs by more than 100 micrometres')
            summary['pairing'][method+'_max_bpm_replay_difference_m']=max(replay_errors,default=0.)
            if method=='bo':
                branch_reference=clean(dict(obs=o,info=i))
            else:
                difference=max(float(np.max(np.abs(np.asarray(o[key])-branch_reference['obs'][key]))) for key in ('bpm_x','bpm_y'))
                summary['pairing']['branch_start_max_bpm_difference_m']=difference
                summary['pairing']['verified_identical_branch_start']=difference==0.
                if difference>1e-4:raise RuntimeError('Paired branch starts do not agree')
            branch_timings['environment_reset_and_replay_s']=time.perf_counter()-branch_started
            stage_started=time.perf_counter()
            before=machine.last[1];print(f'{method}: optics matching',flush=True)
            sized=optics_stage(machine,models,beta,config,method,out)
            branch_timings['optics_s']=time.perf_counter()-stage_started
            stage_started=time.perf_counter()
            if sized['on_screen']:
                print(f'{method}: final corrector centering',flush=True);final=center_stage(machine,models,config,method,out);status='completed' if final['on_screen'] else 'final_readout_lost'
            else:final=sized;status='optics_readout_lost'
            branch_timings['centering_s']=time.perf_counter()-stage_started
            branch_timings['total_s']=time.perf_counter()-branch_started
            summary['branches'][method]=dict(status=status,before=before,after_optics=sized,final=final,wall_time=branch_timings)
    else:summary['status']='threading_budget_exhausted'
    timings['total_s']=time.perf_counter()-run_started;summary['wall_time']=timings
    (out/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+'\n')
    machine.env.close()
    for name,result in summary['branches'].items():
        f=result['final'];print(f"{name}: {result['status']}; transmission={f['transmission']:.3f}, centroid=({f['mu_x']*1e3:.3f}, {f['mu_y']*1e3:.3f}) mm, sigma=({f['sigma_x']*1e3:.3f}, {f['sigma_y']*1e3:.3f}) mm",flush=True)
    return summary
