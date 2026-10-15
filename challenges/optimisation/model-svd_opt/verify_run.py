"""Audit saved controls, stochastic observations, and shot accounting."""
import argparse
import json
from pathlib import Path
import numpy as np

def verify(directory):
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text());rows=[json.loads(s) for s in (directory/'shots.jsonl').read_text().splitlines()]
    if not summary['threading']['success']:return dict(status='threading_failed',shots=len(rows))
    for method in ('bo','rcds'):
        optics=json.loads((directory/f'{method}_optics.json').read_text());space=optics['space'];basis=np.array(space['basis']);base=np.array(space['base']);names=space['names']
        assert basis.shape[1]<=5
        active=[r for r in rows if r['branch']==method and r['stage']=='optics']
        for row in active:
            setting=np.array([row['info']['magnets'][n] for n in names]);delta=setting-base
            # Float32 readbacks have rounding, including a rounded baseline.
            np.testing.assert_allclose(basis@np.linalg.lstsq(basis,delta,rcond=None)[0],delta,atol=4e-6,rtol=1e-5)
        center=[r for r in rows if r['branch']==method and r['stage']=='center']
        allowed={f'cla_feh_mag_hvcor_0{i}_{p}' for i in (1,2) for p in ('h','v')}
        for row in center:assert set(row['action']['magnets'])==allowed
        # Repeated measurements retain genuine shot stochasticity.
        first=active[:summary['config']['repeats']]
        if summary['config']['beam_variation']*summary['config']['beam_jitter'] or summary['config']['ripple']:
            assert any(not np.array_equal(first[0]['obs']['bpm_x'],r['obs']['bpm_x']) for r in first[1:]) or len(first)==1
    count=0
    for row in rows:
        if row['stage']=='reset':count=0
        else:count+=1
        assert row['info']['evaluations']==count
    return dict(status='passed',shots_including_resets=len(rows),pairing=summary['pairing'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');print(json.dumps(verify(p.parse_args().directory),indent=2))
