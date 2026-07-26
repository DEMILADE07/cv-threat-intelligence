from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from customization import CandidateAlert
from verification_gate import VerificationResult, _apply_rule_consistency_guard, _build_question, _question_for_alert, _save_artifacts


def _alert() -> CandidateAlert:
    return CandidateAlert(
        rule_name="shoplifting",
        priority="high",
        detector="concealment",
        title="POSSIBLE CONCEALMENT",
        person_id=1,
        object_label=None,
        timestamp=0.0,
    )


def _result() -> VerificationResult:
    return VerificationResult(
        confirmed=False,
        confidence=0.0,
        reason="test",
        alert_priority="high",
        timestamp="2026-07-04T00:00:00Z",
    )


class VerificationGateArtifactTests(unittest.TestCase):
    def test_build_question_is_rule_specific_for_runtime_rule_names(self) -> None:
        self.assertIn("physical violence or assault", _build_question("violence", "retail_shop"))
        self.assertIn("weak temporal", _build_question("video_action_violence_candidate", "retail_shop"))
        self.assertIn("weapon", _build_question("video_action_weapon_handling_candidate", "retail_shop"))
        self.assertIn("perimeter", _build_question("perimeter_intrusion", "estate_gate"))
        self.assertIn("gate/door", _build_question("tailgating", "estate_gate"))
        self.assertIn("unattended", _build_question("abandoned_object", "estate_gate"))
        self.assertIn("crowd", _build_question("crowd_formation", "estate_gate"))

    def test_gate_prompt_uses_candidate_custom_question(self) -> None:
        alert = _alert()
        alert.question = "Do these frames show the custom compound event?"

        self.assertEqual(
            _question_for_alert(alert, "retail_shop"),
            "Do these frames show the custom compound event?",
        )

    def test_rule_consistency_guard_rejects_theft_reason_for_violence_alert(self) -> None:
        alert = CandidateAlert(
            rule_name="violence",
            priority="critical",
            detector="violence",
            title="VIOLENCE SUSPECTED",
            person_id=None,
            object_label=None,
            timestamp=0.0,
        )
        result = VerificationResult(
            confirmed=True,
            confidence=0.95,
            reason="The frame clearly shows a woman taking merchandise from the shelves.",
            alert_priority="critical",
            timestamp="2026-07-12T00:00:00Z",
        )

        guarded = _apply_rule_consistency_guard(alert, result)

        self.assertFalse(guarded.confirmed)
        self.assertLessEqual(guarded.confidence, 0.2)
        self.assertIn("category mismatch", guarded.reason)

    def test_rule_consistency_guard_ignores_echoed_violence_alert_text(self) -> None:
        alert = CandidateAlert(
            rule_name="violence",
            priority="critical",
            detector="violence",
            title="VIOLENCE SUSPECTED",
            person_id=None,
            object_label=None,
            timestamp=0.0,
        )
        result = VerificationResult(
            confirmed=True,
            confidence=0.95,
            reason="The image clearly shows a woman stealing merchandise, which aligns with the violence suspected alert.",
            alert_priority="critical",
            timestamp="2026-07-12T00:00:00Z",
        )

        guarded = _apply_rule_consistency_guard(alert, result)

        self.assertFalse(guarded.confirmed)
        self.assertIn("category mismatch", guarded.reason)

    def test_save_artifacts_removes_stale_frame_files(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            _save_artifacts(tmp_path, 1, [frame, frame, frame], _alert(), _result(), "{}")
            self.assertTrue((tmp_path / "gate_0001" / "frame_0.jpg").exists())
            self.assertTrue((tmp_path / "gate_0001" / "frame_1.jpg").exists())
            self.assertTrue((tmp_path / "gate_0001" / "frame_2.jpg").exists())

            _save_artifacts(tmp_path, 1, [frame], _alert(), _result(), "{}")

            self.assertTrue((tmp_path / "gate_0001" / "frame.jpg").exists())
            self.assertFalse((tmp_path / "gate_0001" / "frame_0.jpg").exists())
            self.assertFalse((tmp_path / "gate_0001" / "frame_1.jpg").exists())
            self.assertFalse((tmp_path / "gate_0001" / "frame_2.jpg").exists())


if __name__ == "__main__":
    unittest.main()
