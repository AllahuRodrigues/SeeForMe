"""M5: own chest-mounted recordings: narration keywords become timed labels; heads of people are blurred."""
import numpy as np

from seeforme.own.ingest import NARRATION, blur_regions, head_region, parse_narration


def w(word, t):
    return {"word": word, "start": t, "end": t + 0.3}


def test_multiword_phrases_win_over_single_words():
    ev = parse_narration([w("stairs", 3.0), w("down", 3.3), w("door", 9.0), w("clear", 15.0)])
    assert [(e["label"], e["detail"], round(e["t"], 1)) for e in ev] == [("stairs", "down", 3.0), ("door", "", 9.0), ("clear", "", 15.0)]


def test_unrelated_speech_is_ignored_and_case_insensitive():
    ev = parse_narration([w("hello", 1.0), w("Curb", 2.0), w("BIKE", 4.0), w("left", 4.3)])
    assert [(e["label"], e["detail"]) for e in ev] == [("curb", ""), ("bicycle", "left")]


def test_vocabulary_covers_the_protocol():
    labels = {v[0] for v in NARRATION.values()}
    assert {"stairs", "door", "curb", "crossing", "clear", "bicycle", "pole", "elevator"} <= labels


def test_head_region_is_the_top_of_a_person_box_clamped():
    x1, y1, x2, y2 = head_region((0.4, 0.0, 0.6, 0.8))
    assert y1 == 0.0 and y2 < 0.4 and x1 < 0.4 and x2 > 0.6


def test_blur_changes_only_the_region():
    img = np.random.default_rng(0).integers(0, 255, (100, 200, 3), dtype=np.uint8)
    out = blur_regions(img, [(0.0, 0.0, 0.5, 0.5)])
    assert not np.array_equal(out[:50, :100], img[:50, :100])
    assert np.array_equal(out[60:, 120:], img[60:, 120:])


def test_development_and_lockbox_recordings_live_in_separate_folders(tmp_path):
    import pytest
    from seeforme.own.ingest import BATCHES, batch_dirs
    assert BATCHES == ("dev", "lockbox")
    dev_raw, dev_out = batch_dirs(tmp_path, "dev")
    lock_raw, lock_out = batch_dirs(tmp_path, "lockbox")
    assert dev_raw != lock_raw and dev_out != lock_out
    assert lock_raw.name == "lockbox" and lock_out.name == "lockbox"
    assert dev_raw.parent == lock_raw.parent == tmp_path / "data" / "own" / "raw"
    with pytest.raises(ValueError):
        batch_dirs(tmp_path, "test")
