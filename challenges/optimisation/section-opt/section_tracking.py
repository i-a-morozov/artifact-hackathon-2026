"""Reusable particle transport for local moment readouts, without camera side effects."""
import cheetah
import torch


class MomentTracker:
    """Reuse a section and omit nonblocking, zero-length screen diagnostics.

    Retains actual magnets, apertures, and active blocking screens. The local RL
    observation already reads exit particle moments, not Screen.reading. Omitting
    nonblocking screens removes redundant particle clones and lets Cheetah merge
    adjacent linear maps. This tracker must not be used to generate camera images.
    Elements are shared with the lattice so knob updates remain live.
    """
    def __init__(self, lattice, plan):
        elements = []
        for element in lattice.elements[plan.start_index+1:plan.stop_index+1]:
            if (isinstance(element, cheetah.Screen)
                    and not (element.is_active and element.is_blocking)
                    and bool(torch.all(element.length == 0))):
                continue
            elements.append(element)
        self.segment = cheetah.Segment(elements)

    def __call__(self, beam):
        with torch.no_grad():
            return self.segment.track(beam)
