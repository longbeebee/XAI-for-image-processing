"""Adapter for common CelebA-Spoof 44-value annotation vectors."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .metadata import empty_record
from .paths import normalize_relative_path

SPOOF_TYPES = {
    0: "live",
    1: "photo",
    2: "poster",
    3: "A4",
    4: "face_mask",
    5: "upper_body_mask",
    6: "region_mask",
    7: "PC",
    8: "pad",
    9: "phone",
    10: "three_d_mask",
}
ILLUMINATIONS = {0: "live", 1: "normal", 2: "strong", 3: "back", 4: "dark"}
ENVIRONMENTS = {0: "live", 1: "indoor", 2: "outdoor"}


class CelebASpoofAdapter:
    """Parse records while validating label semantics against spoof type."""

    def __init__(self, label_mapping: dict[int, int] | None = None) -> None:
        self.label_mapping = label_mapping

    @staticmethod
    def infer_label_mapping(vectors: Iterable[list[Any]]) -> dict[int, int]:
        """Infer raw index-43 to canonical label by maximizing consistency."""
        observations: Counter[tuple[int, int]] = Counter()
        for vector in vectors:
            if len(vector) >= 44:
                observations[(int(vector[43]), int(int(vector[40]) > 0))] += 1
        raw_values = {raw for raw, _ in observations}
        if not raw_values:
            raise ValueError("No valid 44-element annotations were found.")
        mapping: dict[int, int] = {}
        for raw in raw_values:
            mapping[raw] = max((0, 1), key=lambda canonical: observations[(raw, canonical)])
        return mapping

    def parse(
        self,
        relative_path: str,
        annotation: Any,
        source_split: str,
        subject_id: str | None = None,
    ) -> dict[str, Any]:
        """Parse one annotation into the canonical metadata schema."""
        path = normalize_relative_path(relative_path)
        record = empty_record(path, source_split)
        record["subject_id"] = subject_id
        if not isinstance(annotation, (list, tuple)):
            record["error_message"] = "annotation_not_vector"
            return record
        record["annotation_length"] = len(annotation)
        if len(annotation) < 44:
            record["error_message"] = "annotation_too_short"
            return record
        try:
            spoof_type_id = int(annotation[40])
            illumination_id = int(annotation[41])
            environment_id = int(annotation[42])
            raw_label = int(annotation[43])
            expected_label = int(spoof_type_id > 0)
            mapping = self.label_mapping or {0: 0, 1: 1}
            mapped_label = mapping.get(raw_label)
            consistent = mapped_label == expected_label
            record.update(
                {
                    "label": "spoof" if mapped_label == 1 else "real",
                    "label_id": mapped_label,
                    "spoof_type_id": spoof_type_id,
                    "spoof_type": SPOOF_TYPES.get(spoof_type_id, f"unknown_{spoof_type_id}"),
                    "illumination_id": illumination_id,
                    "illumination": ILLUMINATIONS.get(
                        illumination_id, f"unknown_{illumination_id}"
                    ),
                    "environment_id": environment_id,
                    "environment": ENVIRONMENTS.get(
                        environment_id, f"unknown_{environment_id}"
                    ),
                    "annotation_consistent": consistent,
                    "is_valid": mapped_label in {0, 1} and consistent,
                    "error_message": None if consistent else "label_spoof_type_conflict",
                }
            )
        except (TypeError, ValueError, OverflowError) as exc:
            record["error_message"] = f"annotation_value_error:{type(exc).__name__}"
        return record

