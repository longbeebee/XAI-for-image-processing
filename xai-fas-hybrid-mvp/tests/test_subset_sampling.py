from __future__ import annotations

import pandas as pd

from src.datasets.subset import create_subsets, validate_no_overlap


def frame() -> pd.DataFrame:
    rows = []
    for index in range(40):
        split = "test" if index >= 32 else "train"
        rows.append(
            {
                "image_id": f"id-{index}",
                "relative_path": f"Data/{split}/{index}.jpg",
                "source_split": split,
                "subject_id": None,
                "label": "spoof" if index % 2 else "real",
                "label_id": index % 2,
                "spoof_type": "photo" if index % 2 else "live",
                "annotation_consistent": True,
                "is_valid": True,
            }
        )
    return pd.DataFrame(rows)


def test_same_seed_same_subset_and_no_overlap() -> None:
    config = {"train_samples": 20, "val_samples": 6, "test_samples": 6, "seed": 42}
    first, second = create_subsets(frame(), config), create_subsets(frame(), config)
    assert first["train"]["image_id"].tolist() == second["train"]["image_id"].tolist()
    validate_no_overlap(first)
    assert all("/test/" not in path for path in first["train"]["relative_path"])

