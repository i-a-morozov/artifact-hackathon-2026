#!/usr/bin/env python3
"""Run paired stochastic singular-mode optimization episodes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/clara-matplotlib')
os.environ.setdefault('XSUITE_ALLOW_KERNEL_COMPILATION','1')
import sys
from pathlib import Path
import importlib.util
import argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('model_svd_opt',Path(__file__).parent/'__init__.py',submodule_search_locations=[str(Path(__file__).parent)])
pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
from model_svd_opt.workflow import Config,run
from model_svd_opt.report import plot
import torch
import numpy as np

def main():
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--target-mm',type=float,nargs=2,default=[1.,1.]);p.add_argument('--backend',choices=['cheetah','xsuite'],default='cheetah')
    p.add_argument('--episodes',type=int,default=1);p.add_argument('--sample-targets',action='store_true',help='Draw episode targets log-uniformly within the full 0.1–20 mm calibrated domain')
    p.add_argument('--particles',type=int,default=10000);p.add_argument('--repeats',type=int,default=3)
    for name,default in [('thread-budget',40),('optics-budget',35),('center-budget',25)]:p.add_argument('--'+name,type=int,default=default)
    for name,default in [('beam-variation',1.),('misalignment-variation',1.),('beam-jitter',.1),('ripple',.001),('bpm-noise',0.)]:p.add_argument('--'+name,type=float,default=default)
    a=vars(p.parse_args());out=Path(a.pop('output'));episodes=a.pop('episodes');sample=a.pop('sample_targets');a['target']=tuple(v*1e-3 for v in a.pop('target_mm'))
    if episodes<1:p.error('--episodes must be positive')
    rng=np.random.default_rng(np.random.SeedSequence([a['seed'],917]))
    for episode in range(episodes):
        cfg=dict(a,seed=a['seed']+episode)
        if sample:cfg['target']=tuple(np.exp(rng.uniform(np.log(.0001),np.log(.02),2)))
        destination=out if episodes==1 else out/f'episode_{episode:03d}'
        run(Config(**cfg),destination);plot(destination)

if __name__=='__main__':main()
