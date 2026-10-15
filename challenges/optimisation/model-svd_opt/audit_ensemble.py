#!/usr/bin/env python3
"""Audit the collected episode data, score means, clocks and acquisition counts."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
from run_ensemble import counts,atomic_json
from verify_run import verify

def audit(directory):
    directory=Path(directory).resolve();metrics=json.loads((directory/'metrics.json').read_text());meta=metrics['metadata'];n=meta['requested_episodes']
    assert len(metrics['episodes'])==n
    assert sorted(e['episode'] for e in metrics['episodes'])==list(range(n))
    assert sorted(e['seed'] for e in metrics['episodes'])==list(range(meta['seed_start'],meta['seed_start']+n))
    total_steps=0;total_resets=0;changes=0;failures=[]
    for e in metrics['episodes']:
        path=directory/f"episode_{e['episode']:03d}"
        result=json.loads((path/'episode.json').read_text());rows=result['data']
        assert counts(rows)==result['counts']==e['counts']
        for r in rows:assert isinstance(r['wall_time_s'],(int,float)) and r['wall_time_s']>=0 and math.isfinite(r['wall_time_s'])
        total_steps+=e['counts']['steps'];total_resets+=e['counts']['reset_shots'];changes+=e['counts']['screen_changes']
        if result['status']=='error':
            failures.append(dict(episode=e['episode'],error=e['error']))
        for method,branch in result['branches'].items():
            shots=[r for r in rows if r['branch']==method]
            assert branch['counts']==counts(shots)
            if branch['final']:
                assert branch['step_counter']==shots[-1]['info']['evaluations']==branch['counts']['steps']
                assert branch['last_shot_score']==shots[-1]['info']['benchmark_score']
                if 'benchmark_score_mean' in branch['final']:
                    repeats=result['config']['repeats'];scores=[r['info']['benchmark_score'] for r in shots[-repeats:]]
                    assert math.isclose(branch['final_score'],sum(scores)/len(scores),rel_tol=1e-12,abs_tol=1e-12)
        if result['status']!='error':verify(path)
    aggregate=metrics['aggregate'];assert total_steps==aggregate['total_steps'];assert total_resets==aggregate['total_reset_shots'];assert changes==aggregate['total_screen_changes']
    digest=hashlib.sha256()
    with (directory/'results.json').open('rb') as f:
        while chunk:=f.read(1024*1024):digest.update(chunk)
    report=dict(status='passed',episodes=n,total_steps=total_steps,total_reset_shots=total_resets,total_screen_changes=changes,
        retained_error_outcomes=failures,results_json_sha256=digest.hexdigest(),results_json_bytes=(directory/'results.json').stat().st_size,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),checks=['episode and seed inventory','recorded step/reset/screen-change counts','per-shot finite wall times','native final step counters','single-shot benchmark scores','fresh-verification mean benchmark scores','modal membership and centering restriction','fresh stochastic repeats','workflow shot-counter accounting'])
    atomic_json(directory/'audit.json',report);print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');audit(p.parse_args().directory)
