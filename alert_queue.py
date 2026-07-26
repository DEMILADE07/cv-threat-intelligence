"""Small Verification Gate queue for candidate alerts.

The detector can produce several plausible alerts for the same moment. This helper
lets the VLM reject a wrong high-priority candidate and continue to the next one,
without re-verifying the same candidate every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from customization import CandidateAlert
from verification_gate import VerificationResult


@dataclass(frozen=True)
class GateAttempt:
    signature: str
    alert: CandidateAlert
    result: VerificationResult


def alert_signature(alert: CandidateAlert) -> str:
    return f"{alert.rule_name}:{alert.title}"


def should_log_rejection(rejection_count: int, *, limit: int, force: bool = False) -> bool:
    if force:
        return True
    if limit <= 0:
        return False
    return rejection_count < limit


def verify_alert_queue(
    alerts: list[CandidateAlert],
    *,
    verify: Callable[[CandidateAlert], VerificationResult],
    last_attempts: dict[str, float],
    now: float,
    candidate_limit: int = 3,
    repeat_seconds: float = 2.0,
) -> tuple[CandidateAlert | None, VerificationResult | None, list[GateAttempt]]:
    """Verify a short queue of candidate alerts until one is confirmed.

    `alerts` should already be sorted by the CustomizationEngine. We try only the
    first `candidate_limit` candidates, skip candidates recently checked, and stop
    at the first confirmed result.
    """
    attempts: list[GateAttempt] = []
    limit = max(1, candidate_limit)
    repeat = max(0.0, repeat_seconds)

    for alert in alerts[:limit]:
        signature = alert_signature(alert)
        if now - last_attempts.get(signature, -1_000_000.0) < repeat:
            continue

        result = verify(alert)
        last_attempts[signature] = now
        attempts.append(GateAttempt(signature=signature, alert=alert, result=result))
        if result.confirmed:
            return alert, result, attempts

    return None, None, attempts
