import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collect_ensemble import recover_completed_branches

class ExportRecoveryTest(unittest.TestCase):
    def test_completed_bo_survives_later_rcds_exception(self):
        rows=[]
        for score in (1.,3.):
            info=dict(on_screen=True,transmission=.98,on_screen_fraction=.97,mu_x=.0001,mu_y=.0002,sigma_x=.001,sigma_y=.001,
                      benchmark_score=score,scored_mismatch=score-.5,scored_relative_error=score/2)
            rows.append(dict(branch='bo',stage='center',info=info,wall_time_s=.2))
        result=dict(config=dict(repeats=2),error=dict(type='RuntimeError'),data=rows,
                    branches=dict(bo=dict(status='error'),rcds=dict(status='error')))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);(path/'bo_centering.json').write_text('{}')
            recovered=recover_completed_branches(result,path)
        self.assertEqual(recovered['branches']['bo']['status'],'completed')
        self.assertEqual(recovered['branches']['bo']['final_score'],2.)
        self.assertEqual(recovered['branches']['bo']['score_standard_error'],1.)
        self.assertEqual(recovered['branches']['rcds']['status'],'error')
        self.assertEqual(recovered['branches']['bo']['measured_shot_wall_time_s'],.4)

if __name__=='__main__':unittest.main()
