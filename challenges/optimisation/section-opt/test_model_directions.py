"""Conjugacy, orientation, and full-rank completion of model direction bases."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from work.section_response import model_search_directions, direction_intervals
sys.path.insert(0, str(Path(__file__).resolve().parent / 'rcds'))
from rcds.minimizer import Minimizer
from rcds.wrapper import Wrapper


class ModelDirections(unittest.TestCase):
    def model(self):
        J = np.random.default_rng(731).normal(size=(4, 8))
        return SimpleNamespace(response=J, directions=np.linalg.svd(J,full_matrices=True)[2])

    def test_free_subspace_conjugacy_and_orthonormal_completion(self):
        model = self.model()
        start = np.array([.2, 1., .4, .6, 0., .5, .8, .1])
        D = model_search_directions(model, start)
        np.testing.assert_allclose(D @ D.T, np.eye(8), atol=1e-14)
        bound = (start == 0) | (start == 1)
        free_rows = D[np.all(D[:,bound] == 0,axis=1)]
        self.assertEqual(free_rows.shape, (6, 8))
        H = model.response.T @ model.response
        G = free_rows @ H @ free_rows.T
        np.testing.assert_allclose(G, np.diag(np.diag(G)), atol=1e-13)
        self.assertTrue(np.all(np.diff(direction_intervals(start,D),axis=1)>0))
        # Four observable free modes, two bound axes, then two free null modes.
        self.assertTrue(np.all(np.linalg.norm(model.response @ D[:4].T,axis=0)>1e-8))
        np.testing.assert_array_equal(D[4:6],np.eye(8)[bound])
        np.testing.assert_allclose(model.response @ D[6:].T,0.,atol=1e-14)

    def test_interior_is_original_svd_up_to_sign(self):
        model = self.model()
        D = model_search_directions(model,np.full(8,.5))
        np.testing.assert_allclose(np.abs(D @ model.directions.T),np.eye(8),atol=1e-14)

    def test_zero_response_keeps_complete_basis(self):
        model=SimpleNamespace(response=np.zeros((4,3)),directions=np.eye(3))
        D=model_search_directions(model,np.array([0.,.4,.6]))
        np.testing.assert_allclose(D @ D.T,np.eye(3))
        np.testing.assert_array_equal(D[0],[1.,0.,0.])

    def test_package_hessian_helper_returns_conjugate_rows(self):
        H=torch.tensor([[5.,2.,1.],[2.,3.,.7],[1.,.7,2.]],dtype=torch.float64)
        @Wrapper(nk=3,dtype=torch.float64)
        def objective(x):
            return .5*x@H@x,x.new_tensor(1e-8)
        objective()
        minimizer=Minimizer(objective)
        D=minimizer.matrix(knobs=torch.full((3,),.5,dtype=torch.float64))
        G=D@H@D.T
        torch.testing.assert_close(G,torch.diag(G.diag()),atol=1e-12,rtol=1e-12)
        seen=[]
        def line(sf,knobs,value,error,vector,**kwargs):
            seen.append(vector.clone())
            return knobs,value,error
        minimizer.rcds(torch.full((3,),.5,dtype=torch.float64),D,
                       minimize=line,ni=1,termination=False,warning=False)
        torch.testing.assert_close(torch.stack(seen),D)

if __name__=='__main__': unittest.main()
