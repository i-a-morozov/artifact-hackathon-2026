"""Acquisition-order and exact-boundary regressions."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

from work.section_measurement import full_bo_search
from work.beam_tuning import rcds_search


class QuadraticSession:
    def __init__(self, budget=8):
        self.budget = budget
        self.records = []
    @property
    def remaining(self):
        return self.budget - len(self.records)
    def measure(self, u):
        if self.remaining <= 0:
            raise AssertionError('Extra measurement')
        value = float((np.asarray(u)[0] - .2)**2)
        self.records.append(dict(u=np.asarray(u).copy(), value=value))
        return value, 1e-8
    def best(self):
        return min(self.records, key=lambda record: record['value'])


class AcquisitionSchedule(unittest.TestCase):
    def test_full_bo_switches_at_boundary(self):
        # Observe the actual ask() acquisition, not just the metadata we write.
        from skopt import Optimizer
        original = Optimizer.ask
        asked = []
        def recording_ask(optimizer, *args, **kwargs):
            asked.append(optimizer.acq_func)
            return original(optimizer, *args, **kwargs)
        session = QuadraticSession()
        with patch.object(Optimizer, 'ask', recording_ask):
            selected = full_bo_search(session, [.7], switch_at=4, beta=1.0)
        self.assertEqual(asked, ['EI']*3 + ['LCB']*4)
        self.assertEqual([r['acquisition'] for r in session.records],
                         ['start','random','EI','EI','LCB','LCB','LCB','LCB'])
        self.assertEqual(len(session.records), 8)
        np.testing.assert_equal(selected, session.best()['u'])

    def test_switch_does_not_restart_random_initialization(self):
        import warnings
        session = QuadraticSession(12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            full_bo_search(session, [.7]*8, seed=101, switch_at=2)
        probes = np.array([record["u"] for record in session.records])
        self.assertEqual(len(np.unique(probes, axis=0)), len(probes))
        self.assertFalse(any("evaluated at point" in str(w.message) for w in caught))

    def test_full_bo_pure_modes_and_invalid_settings(self):
        for mode, expected in [('ei','EI'),('ucb','LCB')]:
            session = QuadraticSession(4)
            full_bo_search(session, [.7], acquisition=mode)
            self.assertTrue(all(r['acquisition']==expected for r in session.records[1:]))
        session = QuadraticSession()
        for kwargs in [dict(beta=-1),dict(switch_at=9),dict(switch_at=1.5),dict(acquisition='bad')]:
            with self.assertRaises(ValueError): full_bo_search(session,[.7],**kwargs)
        self.assertEqual(len(session.records),0)

    def test_rcds_reuses_acquisitions_in_requested_order(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent / 'rcds'))
        import rcds.minimizer as module
        real_ei, real_ucb = module.LogExpectedImprovement, module.UpperConfidenceBound
        torch.set_num_threads(1)
        for ei_first, expected in [(True,['EI','EI','UCB','UCB']),
                                   (False,['UCB','UCB','EI','EI'])]:
            calls, betas = [], []
            def ei(*args, **kwargs):
                calls.append('EI')
                self.assertFalse(kwargs['maximize'])
                return real_ei(*args, **kwargs)
            def ucb(*args, **kwargs):
                calls.append('UCB')
                betas.append(float(args[1]))
                self.assertFalse(kwargs['maximize'])
                return real_ucb(*args, **kwargs)
            with patch.object(module,'LogExpectedImprovement',ei), patch.object(module,'UpperConfidenceBound',ucb):
                _, history = rcds_search(
                    lambda u: (float((u[0]-.2)**2),1e-5), np.array([.7]),
                    budget=5, method='gp', seed=5, no_ei=2, no_ucb=2,
                    ei_first=ei_first, beta=torch.tensor([1.,4.]),
                    use_parabola=False, nr=2, rs=8,
                )
            self.assertEqual(calls[:4],expected)
            self.assertEqual(betas[:2],[1.,4.])
            self.assertEqual(len(history),5)


if __name__=='__main__': unittest.main()
