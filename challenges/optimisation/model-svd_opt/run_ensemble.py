#!/usr/bin/env python3
"""Run independent stochastic episodes and export one complete JSON dataset."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/clara-ensemble-matplotlib')

def json_default(value):
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'item'):return value.item()
    raise TypeError(f'Cannot encode {type(value).__name__} as JSON')

def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(data,f,allow_nan=False,separators=(',',':'),default=json_default)
    tmp.replace(path)

def read_shots(path):
    if not path.exists():return []
    with path.open() as f:return [json.loads(line) for line in f if line.strip()]

def counts(rows):
    changes=0;previous=None
    for row in rows:
        screen=row['info']['screen']
        if row['stage']=='reset':previous=screen
        else:
            if previous is not None and previous!=screen:changes+=1
            previous=screen
    return dict(steps=sum(r['stage']!='reset' for r in rows),reset_shots=sum(r['stage']=='reset' for r in rows),
                shots_including_resets=len(rows),screen_changes=changes,
                screen_requests=sum(r.get('action') is not None and r['action'].get('screen') is not None for r in rows))

def episode_result(index,seed,config,directory,summary,error,wall):
    rows=read_shots(directory/'shots.jsonl');branches={}
    for method in ('bo','rcds'):
        subset=[r for r in rows if r['branch']==method];saved=summary.get('branches',{}).get(method,{})
        final=saved.get('final',subset[-1]['info'] if subset else None)
        status=saved.get('status','error' if error and subset else 'skipped')
        score=None if final is None else final.get('benchmark_score_mean',final['benchmark_score'])
        evaluations={}
        for stage,filename in [('optics',f'{method}_optics.json'),('centering',f'{method}_centering.json')]:
            p=directory/filename
            evaluations[stage]=len(json.loads(p.read_text())['records']) if p.exists() else None
        branches[method]=dict(status=status,final_score=score,
            score_standard_error=final.get('benchmark_score_standard_error') if final else None,
            last_shot_score=final.get('benchmark_score') if final else None,
            step_counter=final.get('evaluations') if final else None,
            counts=counts(subset),wall_time=saved.get('wall_time'),objective_evaluations=evaluations,final=final)
    return dict(episode=index,seed=seed,config=config,status='error' if error else summary.get('status','finished'),
        error=error,wall_time_s=wall,workflow_wall_time=summary.get('wall_time'),counts=counts(rows),branches=branches,
        summary=summary,data=rows,artifacts_directory=str(directory.relative_to(ROOT)))

def worker(task):
    started=time.perf_counter();index,seed,base,root=task;directory=Path(root)/f'episode_{index:03d}';directory.parent.mkdir(parents=True,exist_ok=True)
    log=directory.parent/f'episode_{index:03d}.log'
    with log.open('w',buffering=1) as f,redirect_stdout(f),redirect_stderr(f):
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0,str(HERE))
        from run_workflow import Config,run
        cfg=Config(**dict(base,seed=seed));summary={};error=None
        try:summary=run(cfg,directory)
        except Exception as exc:
            error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            traceback.print_exc()
        # A workflow exception still represents an attempted episode; preserve
        # every acquired shot and do not silently replace it with a new seed.
        directory.mkdir(exist_ok=True)
        result=episode_result(index,seed,asdict(cfg),directory,summary,error,time.perf_counter()-started)
        atomic_json(directory/'episode.json',result)
    return {k:v for k,v in result.items() if k not in ('data','summary')}

def stats(values):
    values=[v for v in values if v is not None]
    if not values:return dict(n=0)
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                std=statistics.stdev(values) if len(values)>1 else 0.,min=min(values),max=max(values))

def aggregate(episodes):
    result=dict(episodes=len(episodes),episode_status=dict(Counter(e['status'] for e in episodes)),
                total_steps=sum(e['counts']['steps'] for e in episodes),
                total_reset_shots=sum(e['counts']['reset_shots'] for e in episodes),
                total_screen_changes=sum(e['counts']['screen_changes'] for e in episodes),
                episode_wall_time_s=stats([e['wall_time_s'] for e in episodes]),branches={})
    for method in ('bo','rcds'):
        b=[e['branches'][method] for e in episodes]
        result['branches'][method]=dict(status=dict(Counter(v['status'] for v in b)),
            final_score_all_available=stats([v['final_score'] for v in b]),
            final_score_completed=stats([v['final_score'] for v in b if v['status']=='completed']),
            steps=stats([v['counts']['steps'] for v in b]),
            wall_time_s=stats([v['wall_time']['total_s'] for v in b if v['wall_time']]),
            total_screen_changes=sum(v['counts']['screen_changes'] for v in b))
    return result

def provenance():
    paths=[HERE/n for n in ('run_ensemble.py','workflow.py','models.py','controls.py','requirements.lock')]
    paths+=[ROOT/'RL_framework/rl_framework/envs/sim_env.py',ROOT/'section-opt/beam_tuning.py',ROOT/'section-opt/section_measurement.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def write_complete(path,metadata,episodes):
    # Stream per-episode raw data rather than holding the entire dataset in RAM.
    tmp=path.with_suffix('.json.tmp')
    with tmp.open('w') as f:
        f.write('{"metadata":');json.dump(metadata,f,allow_nan=False)
        f.write(',"aggregate":');json.dump(aggregate(episodes),f,allow_nan=False)
        f.write(',"episodes":[')
        for j,e in enumerate(sorted(episodes,key=lambda x:x['episode'])):
            if j:f.write(',')
            source=ROOT/e['artifacts_directory']/'episode.json'
            with source.open() as src:
                while chunk:=src.read(1024*1024):f.write(chunk)
        f.write(']}\n')
    tmp.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=100);p.add_argument('--workers',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--target-mm',nargs=2,type=float,default=[1.,1.]);p.add_argument('--particles',type=int,default=10000);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--thread-budget',type=int,default=40);p.add_argument('--optics-budget',type=int,default=35);p.add_argument('--center-budget',type=int,default=25)
    a=p.parse_args()
    if min(a.episodes,a.workers,a.particles,a.repeats)<1:p.error('Counts must be positive')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    base=dict(target=[v*.001 for v in a.target_mm],backend='cheetah',particles=a.particles,repeats=a.repeats,thread_budget=a.thread_budget,optics_budget=a.optics_budget,center_budget=a.center_budget)
    metadata=dict(schema_version=1,requested_episodes=a.episodes,seed_start=a.seed,workers=a.workers,base_config=base,
        started_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=provenance(),
        score_definition='SimEnv benchmark_score: lower is better; final_score averages fresh final-verification shots when available; last_shot_score is the unaveraged final shot.',
        step_definition='One SimEnv.step per shot; repeats, replay and restoration included; reset shots counted separately. Episode totals include shared threading discovery plus both optimizer branches.',
        screen_change_definition='A change in selected physical screen between consecutive acquired shots within a reset segment; identical screen requests and initial selection at reset do not count.',
        wall_time_definition='Seconds from perf_counter. Episode worker wall time includes first-use imports, calibrations, environment setup, both optimizer branches and data gathering before serialization. Batch wall time is actual elapsed parallel runtime; sums of episode times are not batch elapsed time.',
        data_definition='All acquired shot observations, actions, readbacks, scores and durations; screen images are not exported. Existing simulation outputs do not retain images.')
    atomic_json(out/'metadata.json',metadata);episodes=[];started=time.perf_counter()
    tasks=[(i,a.seed+i,base,str(out)) for i in range(a.episodes)]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,t):t[0] for t in tasks}
        for future in as_completed(futures):
            episode=future.result();episodes.append(episode)
            checkpoint=dict(metadata=metadata,elapsed_wall_time_s=time.perf_counter()-started,aggregate=aggregate(episodes),episodes=sorted(episodes,key=lambda e:e['episode']))
            atomic_json(out/'metrics.json',checkpoint)
            print(f"{len(episodes)}/{a.episodes} episode {episode['episode']:03d}, seed {episode['seed']}, {episode['status']}, steps={episode['counts']['steps']}, wall={episode['wall_time_s']:.1f}s",flush=True)
    metadata['completed_at_utc']=datetime.now(timezone.utc).isoformat();metadata['wall_time_s']=time.perf_counter()-started
    write_complete(out/'results.json',metadata,episodes)
    metadata['wall_time_including_combined_json_export_s']=time.perf_counter()-started
    atomic_json(out/'metrics.json',dict(metadata=metadata,aggregate=aggregate(episodes),episodes=sorted(episodes,key=lambda e:e['episode'])))
    atomic_json(out/'metadata.json',metadata)
    print(json.dumps(dict(output=str(out/'results.json'),aggregate=aggregate(episodes),wall_time_s=metadata['wall_time_including_combined_json_export_s']),indent=2),flush=True)

if __name__=='__main__':main()
