import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('controls',Path(__file__).resolve().parents[1]/'controls.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class ControlsTest(unittest.TestCase):
    def test_dense_modes_preserve_bounds_and_subspace(self):
        rng=np.random.default_rng(2);Q=np.linalg.qr(rng.normal(size=(15,5)))[0];base={str(i):float(v) for i,v in enumerate(rng.uniform(-.9,.9,15))};limits={n:(-1.,1.) for n in base}
        space=c.ModalSpace(list(base),base,Q,limits,np.ones(5)*10)
        for u in [np.zeros(5),np.ones(5),*rng.uniform(size=(300,5))]:
            values=np.array(list(space.settings(u).values()));self.assertTrue(np.all(np.abs(values)<=1+1e-12));delta=values-np.array(list(base.values()));np.testing.assert_allclose(Q@Q.T@delta,delta,atol=1e-12)
        np.testing.assert_allclose(list(space.settings(space.start).values()),list(base.values()),atol=1e-12)
    def test_missing_bpm_is_not_zero_orbit(self):
        info={'transmission':.5,'on_screen':True};obs={'bpm_valid':np.array([1,0]),'bpm_x':np.zeros(2),'bpm_y':np.zeros(2)}
        value=c.objective(obs,info,'thread',np.ones(2),2);obs['bpm_valid'][:]=1
        self.assertGreater(value,c.objective(obs,info,'thread',np.ones(2),2)+99)
    def test_prefix_stops_at_first_missing(self):self.assertEqual(c.prefix([1,1,0,1]),2)
    def test_lost_screen_is_finite_failure(self):
        value=c.objective({},dict(on_screen=False,transmission=0),'optics',np.ones(2));self.assertTrue(np.isfinite(value));self.assertGreaterEqual(value,1000)
    def test_full_prefix_requires_ip_survival(self):
        obs=dict(bpm_valid=np.ones(2),bpm_x=np.zeros(2),bpm_y=np.zeros(2));info=dict(transmission=.9,on_screen=True)
        good=c.objective(obs,info,'thread',np.ones(2),2);info['on_screen']=False
        self.assertEqual(c.objective(obs,info,'thread',np.ones(2),2)-good,100.)
    def test_losing_camera_never_improves_centering(self):
        info=dict(on_screen=True,mu_x=1.,mu_y=-1.,sigma_x=.001,sigma_y=.001,on_screen_fraction=.1,transmission=.1)
        live=c.objective({},info,'center',np.array([.0001,.0001]));info['on_screen']=False
        self.assertLess(live,c.objective({},info,'center',np.array([.0001,.0001])))
    def test_inverse_candidate_obeys_bounds(self):
        space=c.ModalSpace(['a','b'],dict(a=0.,b=0.),np.eye(2),dict(a=(-1,1),b=(-1,1)),[1,1]);u=space.candidate(np.eye(2),[20.,-10.]);self.assertTrue(np.all((u>=0)&(u<=1)));np.testing.assert_allclose(list(space.settings(u).values()),[-1,.5])

if __name__=='__main__':unittest.main()
