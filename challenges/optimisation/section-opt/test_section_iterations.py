"""Regression tests for damped feedback and collapsed RCDS search intervals."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

from work.section_response import iterative_response_search
from work.beam_tuning import rcds_search

sys.path.insert(0, str(Path(__file__).resolve().parent / 'rcds'))
from rcds.minimizer import Minimizer
from rcds.wrapper import Wrapper


class LinearScreenSession:
    def __init__(self, budget):
        self.budget = budget
        self.records = []
        self.plan = SimpleNamespace(knobs=[object() for _ in range(4)])
        self.optimum = np.array([.2, .3, .4, .1])
    @property
    def remaining(self):
        return self.budget - len(self.records)
    def measure(self, u):
        if self.remaining <= 0: raise AssertionError('overspent measurements')
        residual = u - self.optimum
        self.records.append(dict(u=u.copy(), observed=np.array([0., 0., 1., 1.])+residual,
                                 value=float(np.abs(residual).mean())))
        return self.records[-1]['value'], 1e-8


class IterativeSection(unittest.TestCase):
    def test_repeated_tenth_steps_follow_linear_feedback(self):
        session = LinearScreenSession(6)
        model = SimpleNamespace(target=np.array([0.,0.,1.,1.]), scale=np.ones(4), response=np.eye(4))
        start = np.array([.8,.7,.6,.5])
        original_matrix = model.response.copy()
        selected = iterative_response_search(session, model, start, steps=5,
                                             learning_rate=.1, mode='coupled', ridge=0)
        self.assertEqual(len(session.records),6)
        for step, record in enumerate(session.records):
            np.testing.assert_allclose(record['u'], session.optimum + .9**step*(start-session.optimum))
        np.testing.assert_allclose(selected, session.records[-1]['u'])
        np.testing.assert_equal(model.response, original_matrix)

    def test_response_validates_budget_before_measuring(self):
        session = LinearScreenSession(3)
        with self.assertRaises(ValueError):
            iterative_response_search(session,None,np.zeros(4),steps=3)
        with self.assertRaises(ValueError):
            iterative_response_search(session,None,np.zeros(4),steps=2,learning_rate=0)
        self.assertEqual(len(session.records),0)

    def test_response_selects_measured_best_not_last(self):
        session = LinearScreenSession(3)
        original = session.measure
        def measure(u):
            result = original(u)
            if len(session.records)==2: session.records[-1]['value']=0.0
            return result
        session.measure=measure
        model = SimpleNamespace(target=np.array([0.,0.,1.,1.]), scale=np.ones(4), response=np.eye(4))
        best = iterative_response_search(session,model,np.full(4,.8),steps=2,mode='coupled')
        np.testing.assert_equal(best,session.records[1]['u'])

    def test_blocked_lines_take_no_measurements(self):
        calls=[]
        @Wrapper(nk=2, dtype=torch.float64)
        def objective(u):
            calls.append(u.clone())
            return u.square().sum(), u.new_tensor(1e-6)
        objective()
        minimizer=Minimizer(objective)
        knobs=torch.zeros(2,dtype=torch.float64)
        value=knobs.new_tensor(1.)
        error=knobs.new_tensor(1e-6)
        for direction in [knobs.new_tensor([1.,-1.]), torch.zeros_like(knobs)]:
            lb,ub=minimizer.interval(knobs,direction)
            self.assertEqual(float(ub-lb),0.)
            for method in [minimizer.minimize_gp,minimizer.minimize_parabola]:
                result=method(.03,knobs,value,error,direction)
                for actual,expected in zip(result,(knobs,value,error)):
                    torch.testing.assert_close(actual,expected)
        self.assertEqual(calls,[])

    def test_rcds_at_corner_with_rotated_basis(self):
        # First direction is blocked at (0,0), but the second permits motion inward.
        directions=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
        for method in ['gp','parabola']:
            choice,history=rcds_search(
                lambda u:(float(((u-.2)**2).sum()),1e-6),np.zeros(2),budget=8,
                method=method,directions=directions,seed=1,
                **(dict(no_ei=2,no_ucb=2,ei_first=True,nr=2,rs=8,use_parabola=False)
                   if method=='gp' else {}))
            self.assertEqual(len(history),8)
            self.assertTrue(all(np.isfinite(v) for _,v in history))
            self.assertLess(min(v for _,v in history), history[0][1])
            self.assertTrue(np.all((choice>=0)&(choice<=1)))


if __name__=='__main__': unittest.main()
