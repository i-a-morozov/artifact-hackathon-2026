#!/usr/bin/env python3
"""Rebuild complete ensemble exports from independently saved episode artifacts."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import math
import statistics
import time
from run_ensemble import atomic_json,episode_result,aggregate,write_complete,provenance

def recover_completed_branches(result,directory):
    """Retain a completed branch when a later branch raised before summary save."""
    repeats=result['config']['repeats']
    for method,branch in result['branches'].items():
        rows=[r for r in result['data'] if r['branch']==method]
        branch['measured_shot_wall_time_s']=sum(r['wall_time_s'] for r in rows)
        branch['measured_step_wall_time_s']=sum(r['wall_time_s'] for r in rows if r['stage']!='reset')
        if not result['error'] or not (directory/f'{method}_centering.json').exists():continue
        final_rows=rows[-repeats:]
        if len(final_rows)!=repeats or any(r['stage']!='center' for r in final_rows):continue
        samples=[r['info'] for r in final_rows];info=dict(samples[-1])
        info['on_screen']=sum(i['on_screen'] for i in samples)>repeats/2
        for key in ('transmission','on_screen_fraction','mu_x','mu_y','sigma_x','sigma_y'):
            values=[i[key] for i in samples if i['on_screen']] if key in ('mu_x','mu_y','sigma_x','sigma_y') else [i[key] for i in samples]
            info[key]=statistics.mean(values) if values else 0.
        for key in ('benchmark_score','scored_mismatch','scored_relative_error'):
            scores=[i[key] for i in samples];info[key+'_mean']=statistics.mean(scores)
            info[key+'_standard_error']=statistics.stdev(scores)/math.sqrt(repeats) if repeats>1 else 0.
        branch.update(status='completed' if info['on_screen'] else 'final_readout_lost',final=info,
            final_score=info['benchmark_score_mean'],score_standard_error=info['benchmark_score_standard_error'],
            recovered_from_saved_final_verification=True)
    return result

def checkpoint(directory):
    directory=Path(directory).resolve();metadata=json.loads((directory/'metadata.json').read_text())
    previous=directory/'metrics.json';old=json.loads(previous.read_text()) if previous.exists() else {}
    episodes={e['episode']:e for e in old.get('episodes',[])}
    for i in range(metadata['requested_episodes']):
        if i in episodes:continue
        path=directory/f'episode_{i:03d}';summary_path=path/'summary.json';existing=path/'episode.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text());cfg=json.loads((path/'config.json').read_text());tmp=path/'episode.json.tmp'
            match=re.search(r'"wall_time_s":([0-9.eE+\-]+)',tmp.read_text()) if tmp.exists() else None
            wall=float(match[1]) if match else summary['wall_time']['total_s']
            result=episode_result(i,cfg['seed'],cfg,path,summary,None,wall)
        elif existing.exists():result=json.loads(existing.read_text())
        else:continue
        result=recover_completed_branches(result,path)
        episodes[i]={k:v for k,v in result.items() if k not in ('data','summary')}
    ordered=sorted(episodes.values(),key=lambda e:e['episode']);report=dict(metadata=metadata,aggregate=aggregate(ordered),episodes=ordered)
    report['elapsed_wall_time_s']=(datetime.now(timezone.utc)-datetime.fromisoformat(metadata['started_at_utc'])).total_seconds()
    atomic_json(directory/'metrics.json',report)
    print(json.dumps(dict(episodes=len(ordered),status=report['aggregate']['episode_status'],elapsed_wall_time_s=report['elapsed_wall_time_s']),indent=2))
    return report

def collect(directory):
    directory=Path(directory).resolve();metadata=json.loads((directory/'metadata.json').read_text());episodes=[]
    expected=metadata['requested_episodes']
    for i in range(expected):
        path=directory/f'episode_{i:03d}';cfg=json.loads((path/'config.json').read_text())
        summary_path=path/'summary.json'
        if not summary_path.exists():
            existing=path/'episode.json'
            if not existing.exists():raise RuntimeError(f'Episode {i} has not saved an outcome')
            result=json.loads(existing.read_text())
            if result['status']!='error':raise RuntimeError(f'Episode {i} has an incomplete outcome')
            result=recover_completed_branches(result,path)
            atomic_json(existing,result)
            episodes.append({k:v for k,v in result.items() if k not in ('data','summary')})
            continue
        summary=json.loads(summary_path.read_text())
        tmp=path/'episode.json.tmp'
        # Old exporter files can contain an interrupted NumPy metadata value.
        # The monotonic worker duration was already written before that value.
        match=re.search(r'"wall_time_s":([0-9.eE+\-]+)',tmp.read_text()) if tmp.exists() else None
        existing=path/'episode.json'
        previous=json.loads(existing.read_text()) if existing.exists() else {}
        wall=float(match[1]) if match else previous.get('wall_time_s',summary['wall_time']['total_s'])
        result=episode_result(i,cfg['seed'],cfg,path,summary,None,wall)
        result=recover_completed_branches(result,path)
        atomic_json(path/'episode.json',result)
        episodes.append({k:v for k,v in result.items() if k not in ('data','summary')})
    metadata['completed_at_utc']=datetime.now(timezone.utc).isoformat()
    metadata['wall_time_s']=(datetime.fromisoformat(metadata['completed_at_utc'])-datetime.fromisoformat(metadata['started_at_utc'])).total_seconds()
    metadata['wall_time_definition']='Seconds. Per-shot, workflow-stage and worker episode durations use perf_counter. Worker elapsed time is captured before constructing the export. Batch elapsed time is recovered from UTC start/end timestamps and includes exporter repair; the metrics file additionally reports combined-JSON serialization time. Parallel worker times must not be summed to infer batch elapsed time.'
    metadata['export_recovery']=dict(reason='NumPy metadata serialization corrected after numerical tracking; reconstructed from complete workflow summaries and raw shot logs. No seeds or simulation measurements replaced.',source_sha256=provenance(),batch_clock='UTC elapsed timestamps; per-episode and per-stage durations remain perf_counter timings')
    export_started=time.perf_counter();write_complete(directory/'results.json',metadata,episodes)
    metadata['wall_time_including_combined_json_export_s']=metadata['wall_time_s']+time.perf_counter()-export_started
    atomic_json(directory/'metrics.json',dict(metadata=metadata,aggregate=aggregate(episodes),episodes=episodes))
    atomic_json(directory/'metadata.json',metadata)
    print(json.dumps(dict(output=str(directory/'results.json'),aggregate=aggregate(episodes),wall_time_s=metadata['wall_time_including_combined_json_export_s']),indent=2))
    return episodes

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--checkpoint-only',action='store_true');args=p.parse_args()
    (checkpoint if args.checkpoint_only else collect)(args.directory)
