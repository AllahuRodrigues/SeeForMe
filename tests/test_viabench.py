"""VIABench proactive reminders as human labels of when to speak unprompted (adapter only; the data stays local)."""
from collections import Counter

from seeforme.viabench.data import (Reminder, clip_reminders, decision_label, detector_labels, events_from_reminders,
                                    parse_reminders, split_of)

RAW = {"videos/reminder_vqa/3_douyin_2_c/0416.mp4": {
    "video_uid": "3_douyin_2_c/0416", "caption": "x",
    "annotations": [{"qid": 1, "start_time": 0.0, "end_time": 1.73, "task": "Obstacle Alert", "description": "Pedestrians ahead, please be cautious."},
                    {"qid": 4, "start_time": 4.11, "end_time": 7.86, "task": "Staircase Up/Down", "description": "Stairs ahead, going down."},
                    {"qid": 9, "start_time": 130.0, "end_time": 133.0, "task": "Obstacle Alert", "description": "A car is parked on the right."}]}}


def test_descriptions_map_to_detector_and_structure_labels():
    assert detector_labels("Pedestrians ahead, please be cautious.", "Obstacle Alert") == ["person"]
    assert detector_labels("A car is parked on the right.", "Obstacle Alert") == ["car"]
    assert detector_labels("Please be careful, the road is narrow.", "Road Surface Condition") == []     # no 'car' in 'careful'
    assert detector_labels("Stairs ahead, going down.", "Staircase Up/Down") == ["stairs"]
    assert detector_labels("Handrail ahead on the right, please keep to the right.", "Active Avoidance") == ["railing"]
    assert detector_labels("An electric scooter is coming from behind.", "Obstacle Alert") == ["motorcycle"]
    assert detector_labels("There are bicycles and a utility pole on the left.", "Obstacle Alert") == ["bicycle", "pole"]
    assert detector_labels("Go up.", "Stairs Up/Down") == ["stairs"]                                     # the task names it
    assert detector_labels("You have reached the entrance.", "Exit/Entry Recognition") == ["door"]


def test_parse_and_clip_to_the_processed_part_of_the_video():
    rem = parse_reminders(RAW)["videos/reminder_vqa/3_douyin_2_c/0416.mp4"]["reminders"]
    assert [r.qid for r in rem] == [1, 4, 9] and rem[1].labels == ("stairs",)
    assert [r.qid for r in clip_reminders(rem, max_s=100.0)] == [1, 4]            # a reminder past the cache is not a label


def test_events_keep_the_human_window_and_sentence():
    rem = parse_reminders(RAW)["videos/reminder_vqa/3_douyin_2_c/0416.mp4"]["reminders"]
    ev = events_from_reminders(rem)
    assert ev[1].should_speak and (ev[1].earliest_useful_s, ev[1].latest_safe_s) == (4.11, 7.86)
    assert ev[1].ideal_cue == "Stairs ahead, going down." and ev[1].detail["labels"] == ["stairs"] and ev[1].label == "stairs"
    assert ev[0].detail["label_source"] == "viabench_reminder" and ev[0].detail["task"] == "Obstacle Alert"


def test_split_is_stable_per_source_and_close_to_70_20_10():
    keys = [f"videos/reminder_vqa/6_youtube_c/{i:04d}.mp4" for i in range(2000)]
    c = Counter(split_of(k) for k in keys)
    assert split_of(keys[0]) == split_of(keys[0]) and set(c) == {"via_train", "via_dev", "via_test"}
    assert abs(c["via_train"] / 2000 - 0.7) < 0.04 and abs(c["via_dev"] / 2000 - 0.2) < 0.04 and abs(c["via_test"] / 2000 - 0.1) < 0.03


def test_decision_labels_mask_what_the_annotation_cannot_decide():
    rem = [Reminder(1, 10.0, 12.0, "Obstacle Alert", "Pedestrian ahead.", ("person",)),
           Reminder(2, 30.0, 31.0, "Stairs Up/Down", "Stairs ahead.", ("stairs",))]
    assert decision_label(11.0, "person", rem) == 1
    assert decision_label(9.5, "person", rem) == 1                    # a little early still counts (lead)
    assert decision_label(11.0, "car", rem) is None                   # a reminder is active, but not about this object
    assert decision_label(14.0, "person", rem) is None                # just after one: neither clearly useful nor clearly not
    assert decision_label(20.0, "person", rem) == 0                   # annotators marked nothing anywhere near
    assert decision_label(30.5, "person", rem) is None
