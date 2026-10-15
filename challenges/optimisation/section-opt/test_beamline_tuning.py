import unittest

import numpy as np

from score import READOUT_SCREEN, load_lattice, sample_beam
from work.beamline_tuning import (
    collect_screen_moments, make_plan, perturb_controls, sequential_tune,
)


class BeamlineTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_lattice(count_evaluations=False)
        cls.beam = sample_beam(np.random.default_rng(4), num_particles=500, severity=0.2)

    def test_plan_uses_only_intervening_controls(self):
        plans = make_plan(self.reference, output_screen=READOUT_SCREEN, include_rf_phase=True)
        self.assertGreater(len(plans), 10)
        self.assertEqual(plans[0].input_screen, "cla_s02_sim_aper_01")
        self.assertEqual(plans[0].output_screen, "cla_s02_dia_scr_01")
        self.assertEqual(len(plans[0].knobs), 4)
        for plan in plans:
            names_between = {e.name for e in self.reference.elements[plan.start_index + 1:plan.stop_index]}
            self.assertTrue(all(knob.element_name in names_between for knob in plan.knobs))

    def test_short_sequential_run_respects_each_budget(self):
        stop = "cla_s02_dia_scr_03"
        targets = collect_screen_moments(self.reference, self.beam, output_screen=stop)
        hidden = self.reference.clone()
        perturb_controls(hidden, seed=2, fraction=0.01, include_rf_phase=False)
        plans = make_plan(hidden, output_screen=stop, include_rf_phase=False)
        _, _, rows, sessions = sequential_tune(
            hidden, self.beam, targets, plans, method="rcds", budget_per_segment=3, seed=3,
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(session.records) <= 3 for session in sessions))
        self.assertTrue(all(np.isfinite(row["final_truth"]) for row in rows))


if __name__ == "__main__":
    unittest.main()
