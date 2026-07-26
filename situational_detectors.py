"""Lightweight situational candidate detectors for the hybrid threat pipeline.

These detectors are intentionally conservative candidate generators. They do not
declare final threats; they emit RawEvent-compatible signals that the
CustomizationEngine, compound recipes, VideoMAE temporal witness, and VLM gate can
use together.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from customization import RawEvent


@dataclass
class SituationalAssessment:
    detector: str
    active: bool
    title: str
    level: str
    state: str = ""
    person_id: int | None = None
    object_label: str | None = None
    timestamp: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _bbox(person: Any) -> tuple[float, float, float, float]:
    box = getattr(person, "bbox", (0, 0, 0, 0))
    return tuple(float(v) for v in box)  # type: ignore[return-value]


def _track_id(person: Any) -> int | None:
    raw = getattr(person, "track_id", getattr(person, "tracker_id", None))
    return int(raw) if raw is not None else None


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _frame_diagonal(frame_shape: tuple[int, ...] | None) -> float:
    if not frame_shape or len(frame_shape) < 2:
        return 1.0
    h, w = float(frame_shape[0]), float(frame_shape[1])
    return max(1.0, math.hypot(w, h))


def _zone_names(state: Any) -> list[str]:
    return [str(z) for z in getattr(state, "zones", [])]


def _label(item: Any) -> str:
    return str(getattr(item, "label", "")).strip().lower().replace("_", " ").replace("-", " ")


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _matches_any_zone(name: str, keywords: tuple[str, ...]) -> bool:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    return any(keyword in normalized for keyword in keywords)


class RunningPanicDetector:
    """Detects unusually fast tracked-person movement as a weak running/panic signal."""

    def __init__(
        self,
        speed_ratio_per_second: float = 0.45,
        repeat_seconds: float = 1.0,
    ) -> None:
        self.speed_ratio_per_second = speed_ratio_per_second
        self.repeat_seconds = repeat_seconds
        self._history: dict[int, deque[tuple[float, tuple[float, float]]]] = {}
        self._last_emit: dict[int, float] = {}

    def update(
        self,
        people: Iterable[Any],
        timestamp: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[SituationalAssessment]:
        diagonal = _frame_diagonal(frame_shape)
        assessments: list[SituationalAssessment] = []

        for person in people:
            tid = _track_id(person)
            if tid is None:
                continue
            center = _center(_bbox(person))
            history = self._history.setdefault(tid, deque(maxlen=6))
            if history:
                previous_ts, previous_center = history[0]
                dt = max(timestamp - previous_ts, 1e-6)
                distance = math.hypot(center[0] - previous_center[0], center[1] - previous_center[1])
                speed_ratio = (distance / diagonal) / dt
                last_emit = self._last_emit.get(tid, -1e9)
                if speed_ratio >= self.speed_ratio_per_second and timestamp - last_emit >= self.repeat_seconds:
                    assessments.append(
                        SituationalAssessment(
                            detector="running",
                            active=True,
                            title="RUNNING / PANIC MOVEMENT CANDIDATE",
                            level="medium",
                            state="RUNNING",
                            person_id=tid,
                            timestamp=timestamp,
                            extra={
                                "speed_ratio_per_second": round(speed_ratio, 3),
                                "note": "Fast movement candidate; requires context/VLM confirmation.",
                            },
                        )
                    )
                    self._last_emit[tid] = timestamp
            history.append((timestamp, center))

        return assessments


class PersonDownDetector:
    """Detects a person-shaped box that is unusually horizontal."""

    def __init__(
        self,
        horizontal_ratio: float = 1.25,
        min_area_ratio: float = 0.005,
        repeat_seconds: float = 2.0,
    ) -> None:
        self.horizontal_ratio = horizontal_ratio
        self.min_area_ratio = min_area_ratio
        self.repeat_seconds = repeat_seconds
        self._last_emit: dict[int, float] = {}

    def update(
        self,
        people: Iterable[Any],
        timestamp: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[SituationalAssessment]:
        frame_area = 1.0
        if frame_shape and len(frame_shape) >= 2:
            frame_area = float(max(1, frame_shape[0] * frame_shape[1]))
        assessments: list[SituationalAssessment] = []

        for person in people:
            tid = _track_id(person)
            x1, y1, x2, y2 = _bbox(person)
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            area_ratio = (width * height) / frame_area
            last_key = tid if tid is not None else -1
            last_emit = self._last_emit.get(last_key, -1e9)
            if (
                width / height >= self.horizontal_ratio
                and area_ratio >= self.min_area_ratio
                and timestamp - last_emit >= self.repeat_seconds
            ):
                assessments.append(
                    SituationalAssessment(
                        detector="person_down",
                        active=True,
                        title="PERSON DOWN / FALL CANDIDATE",
                        level="high",
                        state="DOWN",
                        person_id=tid,
                        timestamp=timestamp,
                        extra={
                            "box_aspect_width_over_height": round(width / height, 3),
                            "area_ratio": round(area_ratio, 4),
                        },
                    )
                )
                self._last_emit[last_key] = timestamp

        return assessments


class CameraTamperDetector:
    """Detects blackout, washout, or covered-camera frames."""

    def __init__(
        self,
        dark_mean_threshold: float = 8.0,
        bright_mean_threshold: float = 247.0,
        low_std_threshold: float = 2.0,
        repeat_seconds: float = 3.0,
    ) -> None:
        self.dark_mean_threshold = dark_mean_threshold
        self.bright_mean_threshold = bright_mean_threshold
        self.low_std_threshold = low_std_threshold
        self.repeat_seconds = repeat_seconds
        self._last_emit = -1e9

    def update(self, frame: np.ndarray, timestamp: float) -> list[SituationalAssessment]:
        if frame.size == 0:
            return []
        gray = frame.astype(np.float32).mean(axis=2) if frame.ndim == 3 else frame.astype(np.float32)
        mean = float(gray.mean())
        std = float(gray.std())
        reason = ""
        if mean <= self.dark_mean_threshold:
            reason = "dark/blackout frame"
        elif mean >= self.bright_mean_threshold:
            reason = "over-bright/washed-out frame"
        elif std <= self.low_std_threshold:
            reason = "very low detail/possible lens obstruction"
        if not reason or timestamp - self._last_emit < self.repeat_seconds:
            return []
        self._last_emit = timestamp
        return [
            SituationalAssessment(
                detector="camera_tampering",
                active=True,
                title="CAMERA TAMPERING / OBSTRUCTION CANDIDATE",
                level="high",
                state="TAMPER",
                timestamp=timestamp,
                extra={"reason": reason, "mean": round(mean, 2), "std": round(std, 2)},
            )
        ]


class FireSmokeDetector:
    """Detects broad orange flame-like or gray smoke-like color regions."""

    def __init__(
        self,
        min_fire_ratio: float = 0.025,
        min_smoke_ratio: float = 0.55,
        repeat_seconds: float = 3.0,
    ) -> None:
        self.min_fire_ratio = min_fire_ratio
        self.min_smoke_ratio = min_smoke_ratio
        self.repeat_seconds = repeat_seconds
        self._last_emit = -1e9

    def update(self, frame: np.ndarray, timestamp: float) -> list[SituationalAssessment]:
        if frame.ndim != 3 or frame.size == 0:
            return []
        b = frame[:, :, 0].astype(np.int16)
        g = frame[:, :, 1].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)

        fire_mask = (r > 150) & (g > 70) & (r > g + 35) & (g > b + 35)
        fire_ratio = float(fire_mask.mean())

        max_c = np.maximum.reduce([r, g, b])
        min_c = np.minimum.reduce([r, g, b])
        smoke_mask = (max_c - min_c < 22) & (max_c > 80) & (max_c < 230)
        smoke_ratio = float(smoke_mask.mean())

        if timestamp - self._last_emit < self.repeat_seconds:
            return []
        if fire_ratio >= self.min_fire_ratio:
            state = "FIRE"
            reason = "large orange/red flame-like region"
        elif smoke_ratio >= self.min_smoke_ratio:
            state = "SMOKE"
            reason = "large gray low-saturation smoke-like region"
        else:
            return []

        self._last_emit = timestamp
        return [
            SituationalAssessment(
                detector="fire",
                active=True,
                title="FIRE / SMOKE CANDIDATE",
                level="high",
                state=state,
                timestamp=timestamp,
                extra={
                    "reason": reason,
                    "fire_ratio": round(fire_ratio, 4),
                    "smoke_ratio": round(smoke_ratio, 4),
                },
            )
        ]


class CounterRushDetector:
    """Detects first entry into counter/staff/restricted zones."""

    COUNTER_KEYWORDS = ("counter", "cashier", "till", "staff", "restricted", "vault")

    def __init__(self, repeat_seconds: float = 5.0) -> None:
        self.repeat_seconds = repeat_seconds
        self._previous_zones: dict[int, set[str]] = {}
        self._last_emit: dict[tuple[int, str], float] = {}

    def update(self, zone_states: Iterable[Any], timestamp: float) -> list[SituationalAssessment]:
        assessments: list[SituationalAssessment] = []
        for state in zone_states:
            tid = _track_id(state)
            if tid is None:
                continue
            current_zones = set(_zone_names(state))
            previous_zones = self._previous_zones.get(tid, set())
            entered = current_zones - previous_zones
            for zone in entered:
                if not _matches_any_zone(zone, self.COUNTER_KEYWORDS):
                    continue
                key = (tid, zone)
                last_emit = self._last_emit.get(key, -1e9)
                if timestamp - last_emit < self.repeat_seconds:
                    continue
                assessments.append(
                    SituationalAssessment(
                        detector="counter_rush",
                        active=True,
                        title="COUNTER / RESTRICTED ZONE RUSH CANDIDATE",
                        level="medium",
                        state="ZONE_ENTRY",
                        person_id=tid,
                        timestamp=timestamp,
                        extra={"zone": zone},
                    )
                )
                self._last_emit[key] = timestamp
            self._previous_zones[tid] = current_zones
        return assessments


class TailgatingDetector:
    """Detects multiple tracked people entering an entry/gate zone within a short window."""

    ENTRY_KEYWORDS = ("entrance", "entry", "door", "gate", "lobby", "turnstile")

    def __init__(
        self,
        window_seconds: float = 2.0,
        min_people: int = 2,
        repeat_seconds: float = 5.0,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_people = min_people
        self.repeat_seconds = repeat_seconds
        self._previous_zones: dict[int, set[str]] = {}
        self._entries: dict[str, deque[tuple[float, int]]] = {}
        self._last_emit: dict[str, float] = {}

    def update(self, zone_states: Iterable[Any], timestamp: float) -> list[SituationalAssessment]:
        assessments: list[SituationalAssessment] = []
        for state in zone_states:
            tid = _track_id(state)
            if tid is None:
                continue
            current_zones = set(_zone_names(state))
            previous_zones = self._previous_zones.get(tid, set())
            for zone in current_zones - previous_zones:
                if not _matches_any_zone(zone, self.ENTRY_KEYWORDS):
                    continue
                entries = self._entries.setdefault(zone, deque())
                entries.append((timestamp, tid))
                while entries and timestamp - entries[0][0] > self.window_seconds:
                    entries.popleft()
                recent_ids = {entry_tid for _ts, entry_tid in entries}
                last_emit = self._last_emit.get(zone, -1e9)
                if len(recent_ids) >= self.min_people and timestamp - last_emit >= self.repeat_seconds:
                    assessments.append(
                        SituationalAssessment(
                            detector="tailgating",
                            active=True,
                            title="TAILGATING / MULTI-PERSON ENTRY CANDIDATE",
                            level="high",
                            state="TAILGATING",
                            person_id=tid,
                            timestamp=timestamp,
                            extra={
                                "zone": zone,
                                "people_count": len(recent_ids),
                                "window_seconds": self.window_seconds,
                            },
                        )
                    )
                    self._last_emit[zone] = timestamp
            self._previous_zones[tid] = current_zones
        return assessments


class PerimeterIntrusionDetector:
    """Detects entry into perimeter/fence/boundary zones."""

    PERIMETER_KEYWORDS = ("perimeter", "fence", "boundary", "wall", "restricted_perimeter")

    def __init__(self, repeat_seconds: float = 5.0) -> None:
        self.repeat_seconds = repeat_seconds
        self._previous_zones: dict[int, set[str]] = {}
        self._last_emit: dict[tuple[int, str], float] = {}

    def update(self, zone_states: Iterable[Any], timestamp: float) -> list[SituationalAssessment]:
        assessments: list[SituationalAssessment] = []
        for state in zone_states:
            tid = _track_id(state)
            if tid is None:
                continue
            current_zones = set(_zone_names(state))
            previous_zones = self._previous_zones.get(tid, set())
            for zone in current_zones - previous_zones:
                if not _matches_any_zone(zone, self.PERIMETER_KEYWORDS):
                    continue
                key = (tid, zone)
                last_emit = self._last_emit.get(key, -1e9)
                if timestamp - last_emit < self.repeat_seconds:
                    continue
                assessments.append(
                    SituationalAssessment(
                        detector="perimeter_intrusion",
                        active=True,
                        title="PERIMETER INTRUSION CANDIDATE",
                        level="high",
                        state="INTRUSION",
                        person_id=tid,
                        timestamp=timestamp,
                        extra={"zone": zone},
                    )
                )
                self._last_emit[key] = timestamp
            self._previous_zones[tid] = current_zones
        return assessments


class AbandonedObjectDetector:
    """Detects unattended bags/packages that persist after people move away."""

    OBJECT_LABELS = {
        "backpack",
        "handbag",
        "suitcase",
        "bag",
        "package",
        "box",
        "parcel",
        "duffel bag",
    }

    def __init__(
        self,
        dwell_seconds: float = 10.0,
        owner_distance_ratio: float = 0.18,
        match_distance_ratio: float = 0.08,
        repeat_seconds: float = 10.0,
    ) -> None:
        self.dwell_seconds = dwell_seconds
        self.owner_distance_ratio = owner_distance_ratio
        self.match_distance_ratio = match_distance_ratio
        self.repeat_seconds = repeat_seconds
        self._objects: dict[int, dict[str, Any]] = {}
        self._next_object_id = 1

    def update(
        self,
        detections: Iterable[Any],
        people: Iterable[Any],
        timestamp: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[SituationalAssessment]:
        diagonal = _frame_diagonal(frame_shape)
        person_centers = [_center(_bbox(person)) for person in people]
        candidate_detections = [
            detection for detection in detections
            if _label(detection) in self.OBJECT_LABELS
        ]
        seen_ids: set[int] = set()
        assessments: list[SituationalAssessment] = []

        for detection in candidate_detections:
            label = _label(detection)
            box = _bbox(detection)
            center = _center(box)
            object_id = self._match_object(label, center, diagonal)
            record = self._objects.setdefault(
                object_id,
                {
                    "label": label,
                    "first_seen": timestamp,
                    "last_near_person": timestamp,
                    "last_emit": -1e9,
                    "center": center,
                    "bbox": box,
                },
            )
            record["center"] = center
            record["bbox"] = box
            record["label"] = label
            seen_ids.add(object_id)

            nearest_person_distance = min(
                (_distance(center, person_center) for person_center in person_centers),
                default=float("inf"),
            )
            near_person = nearest_person_distance / diagonal <= self.owner_distance_ratio
            if near_person:
                record["last_near_person"] = timestamp
                continue

            unattended_seconds = timestamp - float(record["last_near_person"])
            if unattended_seconds < self.dwell_seconds:
                continue
            if timestamp - float(record["last_emit"]) < self.repeat_seconds:
                continue
            record["last_emit"] = timestamp
            assessments.append(
                SituationalAssessment(
                    detector="abandoned_object",
                    active=True,
                    title="ABANDONED OBJECT CANDIDATE",
                    level="high",
                    state="UNATTENDED",
                    object_label=label,
                    timestamp=timestamp,
                    extra={
                        "object_id": object_id,
                        "object_label": label,
                        "unattended_seconds": round(unattended_seconds, 2),
                        "bbox": tuple(int(v) for v in box),
                    },
                )
            )

        stale_cutoff = max(self.dwell_seconds * 2.0, 20.0)
        for object_id in list(self._objects):
            if object_id in seen_ids:
                continue
            if timestamp - float(self._objects[object_id].get("first_seen", timestamp)) > stale_cutoff:
                del self._objects[object_id]

        return assessments

    def _match_object(self, label: str, center: tuple[float, float], diagonal: float) -> int:
        threshold = diagonal * self.match_distance_ratio
        for object_id, record in self._objects.items():
            if record.get("label") != label:
                continue
            if _distance(center, record["center"]) <= threshold:
                return object_id
        object_id = self._next_object_id
        self._next_object_id += 1
        return object_id


class CrowdFormationDetector:
    """Detects a close cluster of people as a weak crowd-formation signal."""

    def __init__(
        self,
        min_people: int = 4,
        cluster_radius_ratio: float = 0.18,
        repeat_seconds: float = 5.0,
    ) -> None:
        self.min_people = min_people
        self.cluster_radius_ratio = cluster_radius_ratio
        self.repeat_seconds = repeat_seconds
        self._last_emit = -1e9

    def update(
        self,
        people: Iterable[Any],
        timestamp: float,
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[SituationalAssessment]:
        person_list = list(people)
        if len(person_list) < self.min_people:
            return []
        diagonal = _frame_diagonal(frame_shape)
        threshold = diagonal * self.cluster_radius_ratio
        centers = [(_track_id(person), _center(_bbox(person))) for person in person_list]

        best_cluster: list[int | None] = []
        for _tid, center in centers:
            cluster = [
                other_tid
                for other_tid, other_center in centers
                if _distance(center, other_center) <= threshold
            ]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        if len(best_cluster) < self.min_people or timestamp - self._last_emit < self.repeat_seconds:
            return []
        self._last_emit = timestamp
        return [
            SituationalAssessment(
                detector="crowd_formation",
                active=True,
                title="CROWD FORMATION CANDIDATE",
                level="medium",
                state="CROWD",
                timestamp=timestamp,
                extra={
                    "people_count": len(best_cluster),
                    "track_ids": [tid for tid in best_cluster if tid is not None],
                    "cluster_radius_ratio": self.cluster_radius_ratio,
                },
            )
        ]


class MaskedEntryCandidateDetector:
    """Emits a visual-verification candidate when a person first enters an entry zone."""

    ENTRY_KEYWORDS = ("entrance", "entry", "door", "gate", "lobby")

    def __init__(self, repeat_seconds: float = 10.0) -> None:
        self.repeat_seconds = repeat_seconds
        self._last_emit: dict[tuple[int, str], float] = {}

    def update(self, zone_states: Iterable[Any], timestamp: float) -> list[SituationalAssessment]:
        assessments: list[SituationalAssessment] = []
        for state in zone_states:
            tid = _track_id(state)
            if tid is None:
                continue
            for zone in _zone_names(state):
                if not _matches_any_zone(zone, self.ENTRY_KEYWORDS):
                    continue
                key = (tid, zone)
                last_emit = self._last_emit.get(key, -1e9)
                if timestamp - last_emit < self.repeat_seconds:
                    continue
                assessments.append(
                    SituationalAssessment(
                        detector="masked_entry",
                        active=True,
                        title="MASKED ENTRY CHECK NEEDED",
                        level="low",
                        state="ENTRY",
                        person_id=tid,
                        timestamp=timestamp,
                        extra={
                            "zone": zone,
                            "requires_visual_verification": True,
                            "note": "Zone entry only; VLM must verify whether a mask/face covering is present.",
                        },
                    )
                )
                self._last_emit[key] = timestamp
        return assessments


def situational_to_events(assessments: Iterable[SituationalAssessment]) -> list[RawEvent]:
    return [
        RawEvent(
            detector=assessment.detector,
            active=assessment.active,
            title=assessment.title,
            level=assessment.level,
            state=assessment.state,
            person_id=assessment.person_id,
            object_label=assessment.object_label,
            timestamp=assessment.timestamp,
            extra=assessment.extra,
        )
        for assessment in assessments
    ]
