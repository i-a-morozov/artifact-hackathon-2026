"""Focused tests for the isolated section response workflow."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from score import load_lattice
from work.beam_tuning import sample_beam
from work.beamline_tuning import apply_knobs, beam_moments, track_interval
from work.section_response import (
    build_response_model,
    ideal_kickoff_beam,
    make_section_plan,
    perturb_normalized_start,
    response_candidate,
)


class SectionResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lattice = load_lattice(count_evaluations=False)
        cls.plan = make_section_plan(
            cls.lattice,
            "cla_s04_sim_aper_01",
            "cla_s04_dia_scr_03",
        )
        cls.model = build_response_model(
            cls.lattice,
            cls.plan,
            ideal_kickoff_beam(384),
        )

    def test_s04_plan_and_response_basis(self):
        kinds = [knob.kind for knob in self.plan.knobs]
        self.assertEqual(kinds.count("quadrupole"), 4)
        self.assertEqual(kinds.count("corrector"), 4)
        self.assertEqual(self.model.response.shape, (4, 8))
        self.assertEqual(self.model.directions.shape, (8, 8))
        np.testing.assert_allclose(
            self.model.directions @ self.model.directions.T,
            np.eye(8),
            atol=1e-12,
        )

    def test_response_candidate_is_bounded_and_improves_example(self):
        beam = sample_beam(117, 384)
        start = perturb_normalized_start(self.model.center, seed=17, fraction=0.04)
        apply_knobs(self.lattice, self.plan.knobs, start)
        before = beam_moments(track_interval(self.lattice, self.plan, beam))
        candidate = response_candidate(
            before,
            self.model.target,
            self.model.scale,
            start,
            self.model.response,
            self.plan.knobs,
            mode="block",
        )
        self.assertTrue(np.all((candidate >= 0) & (candidate <= 1)))
        apply_knobs(self.lattice, self.plan.knobs, candidate)
        after = beam_moments(track_interval(self.lattice, self.plan, beam))
        before_value = np.abs((before - self.model.target) / self.model.scale).mean()
        after_value = np.abs((after - self.model.target) / self.model.scale).mean()
        self.assertLess(after_value, before_value)


if __name__ == "__main__":
    unittest.main()
