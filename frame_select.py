"""Per-rule evidence-frame selection for the Verification Gate."""

from __future__ import annotations

import numpy as np


_RULE_FRAMES = {
    "weapon": 1,
    "weapons": 1,
    "gun": 1,
    "knife": 1,
    "violence": 4,
    "assault": 4,
    "fight": 4,
    "concealment": 3,
    "shoplift": 3,
    "theft": 3,
    "robbery": 5,
    "armed_robbery": 5,
    "presence": 1,
    "loiter": 1,
    "zone": 1,
    "after_hours": 1,
    "intrusion": 1,
}
_DEFAULT_FRAMES = 3


def frames_for_rule(rule_name: str) -> int:
    name = (rule_name or "").lower()
    for key, count in _RULE_FRAMES.items():
        if key in name:
            return count
    return _DEFAULT_FRAMES


def _sharpness(frame: np.ndarray) -> float:
    try:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:  # noqa: BLE001 - frame selection must never kill the detector
        return 0.0


def _motion_scores(frames: list[np.ndarray]) -> list[float]:
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return [0.0] * len(frames)

    small = []
    for frame in frames:
        gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        small.append(gray)

    scores = [0.0]
    for idx in range(1, len(small)):
        scores.append(float(np.abs(small[idx] - small[idx - 1]).mean()))
    return scores


def select_evidence_frames(
    recent: list[np.ndarray],
    rule_name: str,
    *,
    count: int | None = None,
) -> tuple[list[np.ndarray], dict]:
    target_count = count if count is not None else frames_for_rule(rule_name)
    if not recent:
        return [], {"strategy": "none", "count": 0, "selected_indices": []}

    if target_count <= 1:
        idx = max(range(len(recent)), key=lambda item: _sharpness(recent[item]))
        return (
            [recent[idx]],
            {
                "strategy": "sharpest",
                "count": 1,
                "selected_indices": [idx],
                "buffer_len": len(recent),
            },
        )

    if len(recent) <= target_count:
        chosen = list(range(len(recent)))
    else:
        last = len(recent) - 1
        chosen = sorted({round(step * last / (target_count - 1)) for step in range(target_count)})

    motion = _motion_scores(recent)
    anchor = max(range(len(recent)), key=lambda item: motion[item])
    return (
        [recent[idx] for idx in chosen],
        {
            "strategy": "motion_peak_span",
            "count": len(chosen),
            "selected_indices": chosen,
            "anchor_index": anchor,
            "buffer_len": len(recent),
        },
    )
