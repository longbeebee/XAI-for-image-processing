from __future__ import annotations

from src.datasets.celeba_spoof_adapter import CelebASpoofAdapter


def make(spoof: int, label: int) -> list[int]:
    values = [0] * 44
    values[40:44] = [spoof, 1, 1, label]
    return values


def test_mapping_is_inferred_when_raw_labels_are_reversed() -> None:
    mapping = CelebASpoofAdapter.infer_label_mapping([make(0, 1), make(2, 0)])
    assert mapping == {1: 0, 0: 1}


def test_canonical_spoof_record() -> None:
    record = CelebASpoofAdapter({0: 0, 1: 1}).parse("Data/train/a.jpg", make(2, 1), "train")
    assert record["label_id"] == 1
    assert record["label"] == "spoof"
    assert record["spoof_type"] == "poster"
    assert record["annotation_consistent"] is True


def test_conflict_and_short_annotations_are_invalid() -> None:
    adapter = CelebASpoofAdapter({0: 0, 1: 1})
    assert not adapter.parse("a.jpg", make(1, 0), "train")["is_valid"]
    assert adapter.parse("a.jpg", [0] * 10, "train")["error_message"] == "annotation_too_short"

