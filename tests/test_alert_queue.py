from __future__ import annotations

import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alert_queue import should_log_rejection, verify_alert_queue
from customization import CandidateAlert
from verification_gate import VerificationResult


def _alert(rule_name: str, priority: str = "high") -> CandidateAlert:
    return CandidateAlert(
        rule_name=rule_name,
        priority=priority,
        detector=rule_name,
        title=rule_name.upper(),
        person_id=None,
        object_label=None,
        timestamp=0.0,
    )


def _result(confirmed: bool) -> VerificationResult:
    return VerificationResult(
        confirmed=confirmed,
        confidence=0.9 if confirmed else 0.2,
        reason="test",
        alert_priority="high",
        timestamp="2026-07-12T00:00:00Z",
    )


class AlertQueueTests(unittest.TestCase):
    def test_should_log_rejection_allows_first_few_by_default(self) -> None:
        self.assertTrue(should_log_rejection(0, limit=3, force=False))
        self.assertTrue(should_log_rejection(2, limit=3, force=False))
        self.assertFalse(should_log_rejection(3, limit=3, force=False))
        self.assertTrue(should_log_rejection(99, limit=3, force=True))

    def test_rejected_top_alert_falls_through_to_next_candidate(self) -> None:
        calls: list[str] = []

        def verify(alert: CandidateAlert) -> VerificationResult:
            calls.append(alert.rule_name)
            return _result(alert.rule_name == "shoplifting")

        selected, result, attempts = verify_alert_queue(
            [_alert("violence", "critical"), _alert("shoplifting", "high")],
            verify=verify,
            last_attempts={},
            now=10.0,
            candidate_limit=3,
            repeat_seconds=2.0,
        )

        self.assertEqual(calls, ["violence", "shoplifting"])
        self.assertEqual(selected.rule_name, "shoplifting")
        self.assertTrue(result.confirmed)
        self.assertEqual([attempt.alert.rule_name for attempt in attempts], ["violence", "shoplifting"])

    def test_recently_attempted_candidate_is_throttled(self) -> None:
        calls: list[str] = []

        def verify(alert: CandidateAlert) -> VerificationResult:
            calls.append(alert.rule_name)
            return _result(True)

        selected, result, attempts = verify_alert_queue(
            [_alert("violence", "critical"), _alert("shoplifting", "high")],
            verify=verify,
            last_attempts={"violence:VIOLENCE": 9.5},
            now=10.0,
            candidate_limit=3,
            repeat_seconds=2.0,
        )

        self.assertEqual(calls, ["shoplifting"])
        self.assertEqual(selected.rule_name, "shoplifting")
        self.assertTrue(result.confirmed)
        self.assertEqual([attempt.alert.rule_name for attempt in attempts], ["shoplifting"])


if __name__ == "__main__":
    unittest.main()
