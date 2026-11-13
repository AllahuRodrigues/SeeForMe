"""Segmentation as an event generator: stairs / door / railing candidates the object detector never produces."""
import numpy as np

from seeforme.perception.seg_candidates import SEG_LABELS, inject_segmentation, seg_detections
from seeforme.types import Detection

GROUPS = ["stairs", "door", "railing", "walkable"]


def seg(ts, stairs, door=None):
    door = door if door is not None else [0.0] * len(ts)
    share = np.array([[s, d, 0.0, 0.5] for s, d in zip(stairs, door)], np.float32)
    return {"t": np.array(ts, np.float32), "share": share, "groups": np.array(GROUPS)}


def test_needs_stable_evidence_over_consecutive_samples():
    z = seg([0, 1, 2, 3], [0.0, 0.10, 0.10, 0.0])
    assert seg_detections(1.5, z, min_share=0.02, min_consecutive=2) == []          # only one sample so far
    dets = seg_detections(2.5, z, min_share=0.02, min_consecutive=2)
    assert [d.label for d in dets] == [SEG_LABELS["stairs"]]
    assert seg_detections(3.5, z, min_share=0.02, min_consecutive=2) == []          # gone


def test_is_causal():
    z = seg([0, 1, 2], [0.10, 0.10, 0.10])
    assert seg_detections(0.99, z, min_share=0.02, min_consecutive=2) == []         # the t=1 sample is in the future


def test_injected_detections_link_to_events_like_detector_output():
    class Ev:
        obj_id, should_speak, earliest_useful_s, latest_safe_s = 7, True, 1.0, 4.0
        detail = {"det_labels": ["stairs"], "t_q": 3.5}
    z = seg([0, 1, 2, 3, 4], [0.1] * 5)
    frames = [(t, [Detection(label="person", confidence=0.9, bbox=(0.1, 0.1, 0.2, 0.3))]) for t in (0.5, 1.5, 2.5, 3.5, 8.0)]
    out = inject_segmentation(frames, z, [Ev()], min_share=0.02, min_consecutive=2)
    seg_dets = [(t, d) for t, ds in out for d in ds if d.label == "stairs"]
    assert {t for t, _ in seg_dets} == {1.5, 2.5, 3.5, 8.0}
    assert all(d.source_id == 7 for t, d in seg_dets if t <= 4.0 + 2.0) and [d.source_id for t, d in seg_dets if t == 8.0] == [None]
    assert all(len(ds) >= 1 and ds[0].label == "person" for _, ds in out)          # detector output untouched


def test_links_events_whose_object_no_detector_could_name():
    # door / stairs events carry the object only in `label` / `concepts`; det_labels and labels are empty
    class Door:
        obj_id, should_speak, label, earliest_useful_s, latest_safe_s = 3, True, "door", 1.0, 4.0
        detail = {"det_labels": [], "labels": [], "concepts": ["door"], "t_q": 3.5}
    z = {"t": np.array([0, 1, 2, 3], np.float32), "share": np.array([[0, 0.1, 0, 0.5]] * 4, np.float32), "groups": np.array(GROUPS)}
    out = inject_segmentation([(t, []) for t in (1.5, 2.5)], z, [Door()], min_share=0.02, min_consecutive=2)
    assert [d.source_id for _, ds in out for d in ds] == [3, 3]
