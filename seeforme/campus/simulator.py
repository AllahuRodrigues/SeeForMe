"""Noisy detector + noisy depth sensor for CampusBench scenarios.

Reuses the detection-noise model of `sim.scenario.DetectorSimulator` (misses, confidence
noise, label confusions, OCR corruption, clutter false positives, low light / blur) and adds
a depth measurement per detection with range-dependent noise, occasional gross errors and a
confidence that falls with range, as a monocular metric-depth model behaves.
"""
from __future__ import annotations

import random
from typing import List, Optional

from ..sim.scenario import DetectorSimulator
from ..types import Detection
from .world import CampusScenario


class CampusDetectorSimulator(DetectorSimulator):
    def __init__(self, scenario: CampusScenario, seed: Optional[int] = None):
        super().__init__(scenario, seed=seed)
        self.sc: CampusScenario = scenario
        self.drng = random.Random((scenario.seed if seed is None else seed) * 7919 + 13)
        self.depth_noise = scenario.depth_noise_rel * (1.5 if scenario.lighting == "low" else 1.0) * (1 + scenario.motion_blur)

    def _measure_depth(self, z: float):
        r = self.drng
        if r.random() < 0.03:
            z_m = z * r.uniform(0.5, 1.8)          # gross error (reflections, texture-less walls)
        else:
            z_m = z * (1 + r.gauss(0, self.depth_noise)) + r.gauss(0, 0.08)
        conf = 0.92 - 0.04 * z + r.gauss(0, 0.05) - (0.15 if self.sc.lighting == "low" else 0.0)
        return max(0.2, z_m), max(0.05, min(0.98, conf))

    def detections_at(self, t: float) -> List[Detection]:
        if self.sc.blacked_out(t):
            return []
        dets = super().detections_at(t)
        by_id = {o.obj_id: o for o in self.sc.objects}
        for d in dets:
            if d.source_id is not None:
                _, z = by_id[d.source_id].rel_pos(t)
                d.distance_m, d.depth_conf = self._measure_depth(z)
            else:
                d.distance_m, d.depth_conf = self.drng.uniform(1.0, 7.0), self.drng.uniform(0.2, 0.5)
        return dets
