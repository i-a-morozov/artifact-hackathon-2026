import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('ensemble',Path(__file__).resolve().parents[1]/'run_ensemble.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

class EnsembleAccountingTest(unittest.TestCase):
    def test_same_screen_requests_are_not_changes(self):
        rows=[dict(stage='reset',info=dict(screen='IP'),action=None)]
        rows.extend(dict(stage='optics',info=dict(screen='IP'),action=dict(screen='IP')) for _ in range(3))
        self.assertEqual(e.counts(rows),dict(steps=3,reset_shots=1,shots_including_resets=4,screen_changes=0,screen_requests=3))
    def test_actual_changes_and_reset_boundaries(self):
        rows=[dict(stage=stage,info=dict(screen=screen),action=None) for stage,screen in [('reset','IP'),('thread','upstream'),('thread','IP'),('reset','upstream'),('thread','upstream')]]
        self.assertEqual(e.counts(rows)['screen_changes'],2)
    def test_empty_statistics_are_explicit(self):self.assertEqual(e.stats([None,None]),dict(n=0))
    def test_numpy_metadata_roundtrip(self):
        import json,tempfile,numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'result.json';e.atomic_json(path,dict(target=np.array([.001,.001]),score=np.float64(1.5)))
            self.assertEqual(json.loads(path.read_text()),dict(target=[.001,.001],score=1.5))
    def test_score_statistics(self):
        result=e.stats([1.,2.,3.,None]);self.assertEqual(result['n'],3);self.assertEqual(result['mean'],2.);self.assertEqual(result['std'],1.)

if __name__=='__main__':unittest.main()
