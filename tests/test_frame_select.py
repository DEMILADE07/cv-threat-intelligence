from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_select import frames_for_rule, select_evidence_frames


class FrameSelectTests(unittest.TestCase):
    def test_frames_for_rule_uses_rule_specific_counts(self) -> None:
        self.assertEqual(frames_for_rule("weapon_sighting"), 1)
        self.assertEqual(frames_for_rule("violence"), 4)
        self.assertEqual(frames_for_rule("shoplifting"), 3)
        self.assertEqual(frames_for_rule("armed_robbery"), 5)
        self.assertEqual(frames_for_rule("after_hours_shelf"), 1)

    def test_multi_frame_rules_span_recent_buffer(self) -> None:
        frames = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(10)]

        selected, meta = select_evidence_frames(frames, "violence")

        self.assertEqual([int(frame[0, 0, 0]) for frame in selected], [0, 3, 6, 9])
        self.assertEqual(meta["strategy"], "motion_peak_span")
        self.assertEqual(meta["selected_indices"], [0, 3, 6, 9])

    def test_single_frame_rules_pick_one_frame(self) -> None:
        frames = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(5)]

        selected, meta = select_evidence_frames(frames, "weapon_sighting")

        self.assertEqual(len(selected), 1)
        self.assertEqual(meta["strategy"], "sharpest")


if __name__ == "__main__":
    unittest.main()
