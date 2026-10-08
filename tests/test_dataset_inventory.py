from scripts.dataset_inventory import inventory


def test_inventory_only_videos_sorted_and_unreviewed(tmp_path):
    (tmp_path / "EgoBlind").mkdir()
    (tmp_path / "JRDB").mkdir()
    (tmp_path / "EgoBlind" / "abc.mp4").write_bytes(b"video1")
    (tmp_path / "JRDB" / "def.MOV").write_bytes(b"video2")
    (tmp_path / "JRDB" / "labels.json").write_text("{}")
    records = list(inventory(tmp_path))
    assert len(records) == 2
    assert {r["source_dataset"] for r in records} == {"EgoBlind", "JRDB"}
    assert all(r["annotation_status"] == "unreviewed" for r in records)
    assert all(r["split"] == "unassigned" for r in records)
