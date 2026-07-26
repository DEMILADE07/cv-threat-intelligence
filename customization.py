"""Customization Engine — evaluates user_config.json rules against Detection Core raw events."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from detector import ThreatAssessment


@dataclass
class RawEvent:
    detector: str           # "weapons" | "violence" | "theft"
    active: bool
    title: str              # e.g. "POSSIBLE THEFT", "VIOLENCE SUSPECTED", "DANGEROUS OBJECT"
    level: str              # "none" | "low" | "high" | "critical"
    state: str = ""         # internal state name, e.g. "DEPART", "ACQUIRE", "APPROACH"
    person_id: int | None = None
    object_label: str | None = None
    timestamp: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class CandidateAlert:
    rule_name: str
    priority: str           # "low" | "medium" | "high" | "critical"
    detector: str
    title: str
    person_id: int | None
    object_label: str | None
    timestamp: float
    reasons: list[str] = field(default_factory=list)
    question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "priority": self.priority,
            "detector": self.detector,
            "title": self.title,
            "person_id": self.person_id,
            "object_label": self.object_label,
            "timestamp": self.timestamp,
            "reasons": self.reasons,
            "question": self.question,
        }


PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


class CustomizationEngine:
    """Evaluates a list of RawEvents against user-defined rules and returns CandidateAlerts."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        baseline_path: str | Path | None = None,
    ) -> None:
        self.rules: list[dict] = []
        self.baseline_rules: list[dict] = []
        self.use_case_id: str = "default"
        if baseline_path:
            self.load_baseline(baseline_path)
        if config_path:
            self.load(config_path)

    def load(self, config_path: str | Path) -> None:
        path = Path(config_path)
        if not path.exists():
            print(f"[CustomizationEngine] Config not found: {path}")
            return
        data = json.loads(path.read_text())
        self.use_case_id = data.get("use_case_id", "default")
        self.rules = data.get("rules", [])
        print(f"[CustomizationEngine] Loaded {len(self.rules)} rules for use-case '{self.use_case_id}'")

    def load_baseline(self, baseline_path: str | Path) -> None:
        path = Path(baseline_path)
        if not path.exists():
            print(f"[CustomizationEngine] Baseline config not found: {path}")
            return
        data = json.loads(path.read_text())
        self.baseline_rules = data.get("rules", [])
        print(f"[CustomizationEngine] Loaded {len(self.baseline_rules)} always-on baseline rule(s)")

    def load_additional(self, config_path: str | Path) -> None:
        path = Path(config_path)
        if not path.exists():
            print(f"[CustomizationEngine] Additional config not found: {path}")
            return
        data = json.loads(path.read_text())
        extra_rules = data.get("rules", [])
        self.rules.extend(extra_rules)
        print(f"[CustomizationEngine] Loaded {len(extra_rules)} additional rule(s) from {path}")

    def evaluate(
        self,
        events: list[RawEvent],
        scene_context: dict | None = None,
        now: datetime | None = None,
    ) -> list[CandidateAlert]:
        """Return all rules that match the current events, sorted highest priority first."""
        now = now or datetime.now()
        context = scene_context or {}
        alerts: list[CandidateAlert] = []

        for rule in self.baseline_rules + self.rules:
            if "signals" in rule:
                compound = _eval_compound(rule, events, now)
                if compound is not None:
                    alerts.append(compound)
                continue
            trigger = rule.get("trigger", {})
            for event in events:
                if not event.active:
                    continue
                if not _match_trigger(event, trigger):
                    continue
                if not _match_time_filter(rule.get("time_filter"), now):
                    continue
                if not _match_context_filter(rule.get("context_filter"), event, context):
                    continue
                alerts.append(
                    CandidateAlert(
                        rule_name=rule["name"],
                        priority=rule.get("priority", "medium"),
                        detector=event.detector,
                        title=event.title,
                        person_id=event.person_id,
                        object_label=event.object_label,
                        timestamp=event.timestamp,
                    )
                )
                break  # one alert per rule per frame is enough

        alerts.sort(key=lambda a: PRIORITY_ORDER.get(a.priority, 0), reverse=True)
        return alerts

    def top_alert(
        self,
        events: list[RawEvent],
        scene_context: dict | None = None,
        now: datetime | None = None,
    ) -> CandidateAlert | None:
        alerts = self.evaluate(events, scene_context, now)
        return alerts[0] if alerts else None


# ---------------------------------------------------------------------------
# Trigger matching
# ---------------------------------------------------------------------------

def _match_trigger(event: RawEvent, trigger: dict) -> bool:
    detector = trigger.get("detector")
    if detector and event.detector != detector:
        return False

    state = trigger.get("state")
    if state:
        # match against explicit state field first, fall back to title substring
        if event.state:
            if state.upper() != event.state.upper():
                return False
        elif state.upper() not in event.title.upper():
            return False

    level = trigger.get("level")
    if level and event.level != level:
        return False

    return True


# ---------------------------------------------------------------------------
# Time filter  "HH:MM-HH:MM"  (supports overnight ranges like 22:00-06:00)
# ---------------------------------------------------------------------------

def _match_time_filter(time_filter: str | None, now: datetime) -> bool:
    if not time_filter:
        return True
    m = re.match(r"(\d{1,2}:\d{2})-(\d{1,2}:\d{2})", time_filter)
    if not m:
        return True
    start = _hhmm_to_minutes(m.group(1))
    end = _hhmm_to_minutes(m.group(2))
    current = now.hour * 60 + now.minute
    if start <= end:
        return start <= current < end
    # overnight: e.g. 22:00–06:00
    return current >= start or current < end


def _hhmm_to_minutes(t: str) -> int:
    h, mn = t.split(":")
    return int(h) * 60 + int(mn)


# ---------------------------------------------------------------------------
# Context filter  — tiny safe expression evaluator
# Supports: ==, !=, >, <, >=, <=, and, or, not, string literals, None
# ---------------------------------------------------------------------------

_ALLOWED_NAMES = {"True", "False", "None", "and", "or", "not", "in"}


def _match_context_filter(
    expr: str | None,
    event: RawEvent,
    context: dict,
) -> bool:
    if not expr:
        return True

    ns: dict[str, Any] = {**context}
    ns.update(
        {
            "detector": event.detector,
            "title": event.title,
            "level": event.level,
            "person_id": event.person_id,
            "object_label": event.object_label,
        }
    )
    ns.update(event.extra)

    try:
        return bool(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
    except Exception:
        return True  # don't block alert if expression is malformed


# ---------------------------------------------------------------------------
# Compound threat recipes
# ---------------------------------------------------------------------------

_SIGNAL_ALIASES = {
    "weapon_candidate": "weapons",
    "weapon": "weapons",
    "gun": "weapons",
    "knife": "weapons",
    "violence": "violence",
    "assault": "violence",
    "fight": "violence",
    "person_down": "person_down",
    "fall": "person_down",
    "concealment": "concealment",
    "theft": "theft",
    "running": "running",
    "panic": "running",
    "masked_entry": "masked_entry",
    "counter_rush": "counter_rush",
    "tailgating": "tailgating",
    "abandoned_object": "abandoned_object",
    "perimeter_intrusion": "perimeter_intrusion",
    "fence_climbing": "perimeter_intrusion",
    "crowd": "crowd_formation",
    "crowd_formation": "crowd_formation",
    "video_action": "video_action",
}


def _match_signal(event: RawEvent, spec: Any) -> bool:
    if isinstance(spec, dict):
        return _match_trigger(event, spec)
    target = _SIGNAL_ALIASES.get(str(spec), str(spec))
    signal_type = str(event.extra.get("signal_type") or "")
    return (
        event.detector == target
        or event.detector == spec
        or str(spec) in signal_type
        or target in signal_type
    )


def _logic_satisfied(logic: str, severities: list[int], total_signals: int) -> bool:
    count = len(severities)
    normalized = (logic or "any").lower()
    if normalized == "all":
        return total_signals > 0 and count >= total_signals
    if normalized == "any":
        return count >= 1
    if normalized.startswith("at_least_"):
        try:
            return count >= int(normalized.rsplit("_", 1)[1])
        except ValueError:
            return count >= 1
    if normalized == "one_high_or_two_medium":
        highs = sum(1 for severity in severities if severity >= PRIORITY_ORDER["high"])
        mediums = sum(1 for severity in severities if severity >= PRIORITY_ORDER["medium"])
        return highs >= 1 or mediums >= 2
    return count >= 1


def _eval_compound(rule: dict, events: list[RawEvent], now: datetime) -> CandidateAlert | None:
    if not _match_time_filter(rule.get("time_filter"), now):
        return None
    specs = rule.get("signals", [])
    present: dict[str, int] = {}
    latest_ts = 0.0
    for event in events:
        if not event.active:
            continue
        for spec in specs:
            if not _match_signal(event, spec):
                continue
            key = spec if isinstance(spec, str) else spec.get("name", str(spec))
            severity = PRIORITY_ORDER.get(event.level, PRIORITY_ORDER["medium"])
            present[str(key)] = max(present.get(str(key), 0), severity)
            latest_ts = max(latest_ts, event.timestamp)
    if not present:
        return None
    if not _logic_satisfied(rule.get("logic", "any"), list(present.values()), len(specs)):
        return None
    return CandidateAlert(
        rule_name=rule["name"],
        priority=rule.get("priority", "high"),
        detector="compound",
        title=rule.get("title", rule["name"].replace("_", " ").upper()),
        person_id=None,
        object_label=None,
        timestamp=latest_ts,
        reasons=[f"{key}={_rank_name(rank)}" for key, rank in present.items()],
        question=rule.get("gate_question"),
    )


def _rank_name(rank: int) -> str:
    for name, value in PRIORITY_ORDER.items():
        if value == rank:
            return name
    return "medium"


# ---------------------------------------------------------------------------
# Converter: ThreatAssessment → RawEvent (used by detector.py)
# ---------------------------------------------------------------------------

def zone_states_to_events(zone_states: list[Any], timestamp: float = 0.0) -> list[RawEvent]:
    """Bridge RetailZoneMonitor output -> 'presence' RawEvents for the Customization Engine.

    This is the wiring that lets rules reason about ZONE + TIME + DWELL — i.e. the GTM
    property rules (loitering, after-hours presence, perimeter intrusion) and examples like
    "anyone in the vault zone after 8pm is a threat". Each (person, occupied-zone) pair
    becomes one active 'presence' event carrying `zone`, `dwell_seconds` and `loitering` in
    `extra`, so a rule's context_filter can read e.g. `zone == 'vault'` or
    `zone == 'aisle' and dwell_seconds >= 8`, and its time_filter can scope it to after hours.

    Accepts any objects exposing `.tracker_id`, `.zones` (list[str]),
    `.dwell_seconds` (dict[str,float]) and `.loitering` (bool) — i.e. retail_zones.PersonZoneState —
    without importing it, to keep this module free of CV dependencies.
    """
    events: list[RawEvent] = []
    for state in zone_states:
        tid = getattr(state, "tracker_id", None)
        zones = getattr(state, "zones", []) or []
        dwell_map = getattr(state, "dwell_seconds", {}) or {}
        loitering = bool(getattr(state, "loitering", False))
        for zone in zones:
            dwell = float(dwell_map.get(zone, 0.0))
            events.append(
                RawEvent(
                    detector="presence",
                    active=True,
                    title=f"PERSON IN ZONE {zone.upper()}",
                    level="low",
                    person_id=tid,
                    timestamp=timestamp,
                    extra={"zone": zone, "dwell_seconds": dwell, "loitering": loitering},
                )
            )
    return events


def concealment_to_events(assessments: list[Any], timestamp: float = 0.0) -> list[RawEvent]:
    """Bridge ConcealmentDetector output -> 'concealment' RawEvents for the engine.

    Only firing candidates become active events (the engine skips inactive ones). Each
    carries the concealment `destination` (waist | bag) and `score` in `extra`, so a rule
    can scope on them if desired. Accepts concealment.ConcealmentAssessment objects
    (duck-typed) without importing them.
    """
    events: list[RawEvent] = []
    for a in assessments:
        destination = getattr(a, "destination", None)
        title = "POSSIBLE CONCEALMENT"
        if destination:
            title += f" ({destination})"
        events.append(
            RawEvent(
                detector="concealment",
                active=bool(getattr(a, "candidate", False)),
                title=title,
                level="high" if getattr(a, "candidate", False) else "none",
                person_id=getattr(a, "track_id", None),
                timestamp=timestamp,
                extra={"destination": destination, "score": float(getattr(a, "score", 0.0))},
            )
        )
    return events


def assessments_to_events(
    object_assessment: ThreatAssessment | None,
    violence_assessment: ThreatAssessment | None,
    theft_assessment: ThreatAssessment | None,
    timestamp: float = 0.0,
    theft_detector: Any = None,
) -> list[RawEvent]:
    events: list[RawEvent] = []

    if object_assessment is not None:
        events.append(
            RawEvent(
                detector="weapons",
                active=object_assessment.active,
                title=object_assessment.title,
                level=object_assessment.level,
                timestamp=timestamp,
                extra={"weapon_labels": object_assessment.weapon_labels},
            )
        )

    if violence_assessment is not None:
        events.append(
            RawEvent(
                detector="violence",
                active=violence_assessment.active,
                title=violence_assessment.title,
                level=violence_assessment.level,
                timestamp=timestamp,
            )
        )

    if theft_assessment is not None:
        # Derive the highest-priority active state across all tracked persons
        active_state = ""
        obj_label = None
        if theft_detector is not None:
            state_priority = {"DEPART": 3, "ACQUIRE": 2, "APPROACH": 1, "IDLE": 0}
            best = -1
            for ps in theft_detector.person_states.values():
                p = state_priority.get(ps.state, 0)
                if p > best:
                    best = p
                    active_state = ps.state
        if not obj_label:
            obj_label = (
                theft_assessment.explicit_labels[0]
                if theft_assessment.explicit_labels
                else None
            )
        events.append(
            RawEvent(
                detector="theft",
                active=theft_assessment.active,
                title=theft_assessment.title,
                level=theft_assessment.level,
                state=active_state,
                timestamp=timestamp,
                object_label=obj_label,
            )
        )

    return events
