from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from situational_detectors import (  # noqa: E402
    AbandonedObjectDetector,
    CameraTamperDetector,
    CounterRushDetector,
    CrowdFormationDetector,
    FireSmokeDetector,
    MaskedEntryCandidateDetector,
    PerimeterIntrusionDetector,
    PersonDownDetector,
    RunningPanicDetector,
    TailgatingDetector,
    situational_to_events,
)


@dataclass
class FakePerson:
    track_id: int
    bbox: tuple[int, int, int, int]


@dataclass
class FakeDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class FakeZoneState:
    tracker_id: int
    bbox: tuple[int, int, int, int]
    zones: list[str] = field(default_factory=list)


class SituationalDetectorTests(unittest.TestCase):
    def test_running_detector_fires_on_fast_tracked_motion(self) -> None:
        detector = RunningPanicDetector(speed_ratio_per_second=0.25)
        detector.update([FakePerson(1, (10, 10, 50, 90))], timestamp=0.0, frame_shape=(100, 100, 3))

        assessments = detector.update(
            [FakePerson(1, (70, 10, 110, 90))],
            timestamp=1.0,
            frame_shape=(100, 100, 3),
        )

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].detector, "running")
        self.assertTrue(assessments[0].active)

    def test_person_down_detector_fires_on_horizontal_body_box(self) -> None:
        detector = PersonDownDetector(horizontal_ratio=1.35)

        assessments = detector.update(
            [FakePerson(8, (10, 60, 90, 100))],
            timestamp=2.0,
            frame_shape=(120, 120, 3),
        )

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].detector, "person_down")
        self.assertEqual(assessments[0].level, "high")

    def test_camera_tamper_detector_fires_on_black_frame(self) -> None:
        detector = CameraTamperDetector(dark_mean_threshold=5.0)
        frame = np.zeros((40, 40, 3), dtype=np.uint8)

        assessments = detector.update(frame, timestamp=3.0)

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].detector, "camera_tampering")
        self.assertIn("dark", assessments[0].extra["reason"])

    def test_fire_smoke_detector_fires_on_large_orange_region(self) -> None:
        detector = FireSmokeDetector(min_fire_ratio=0.2)
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[:, :] = (0, 120, 255)  # BGR orange/red region

        assessments = detector.update(frame, timestamp=4.0)

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].detector, "fire")
        self.assertEqual(assessments[0].level, "high")

    def test_counter_rush_detector_fires_once_when_person_enters_counter_zone(self) -> None:
        detector = CounterRushDetector()
        outside = [FakeZoneState(2, (0, 0, 20, 80), zones=[])]
        inside = [FakeZoneState(2, (0, 0, 20, 80), zones=["cash_counter"])]

        detector.update(outside, timestamp=0.0)
        first = detector.update(inside, timestamp=1.0)
        second = detector.update(inside, timestamp=2.0)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].detector, "counter_rush")
        self.assertEqual(second, [])

    def test_tailgating_detector_fires_when_two_people_enter_gate_together(self) -> None:
        detector = TailgatingDetector(window_seconds=2.0)
        first = [FakeZoneState(1, (0, 0, 20, 80), zones=["front_gate"])]
        second = [FakeZoneState(1, (0, 0, 20, 80), zones=["front_gate"]), FakeZoneState(2, (25, 0, 45, 80), zones=["front_gate"])]

        detector.update(first, timestamp=1.0)
        assessments = detector.update(second, timestamp=2.0)

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].detector, "tailgating")
        self.assertEqual(assessments[0].extra["zone"], "front_gate")

    def test_perimeter_intrusion_detector_fires_on_perimeter_zone_entry(self) -> None:
        detector = PerimeterIntrusionDetector()
        assessments = detector.update(
            [FakeZoneState(3, (0, 0, 20, 80), zones=["north_perimeter_fence"])],
            timestamp=3.0,
        )

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].detector, "perimeter_intrusion")

    def test_abandoned_object_detector_fires_after_object_is_unattended(self) -> None:
        detector = AbandonedObjectDetector(dwell_seconds=5.0, owner_distance_ratio=0.1)
        bag = [FakeDetection("backpack", 0.9, (40, 40, 60, 60))]

        detector.update(bag, people=[FakePerson(1, (42, 20, 62, 80))], timestamp=0.0, frame_shape=(100, 100, 3))
        detector.update(bag, people=[], timestamp=3.0, frame_shape=(100, 100, 3))
        assessments = detector.update(bag, people=[], timestamp=6.0, frame_shape=(100, 100, 3))

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].detector, "abandoned_object")
        self.assertEqual(assessments[0].object_label, "backpack")

    def test_crowd_formation_detector_fires_on_close_group(self) -> None:
        detector = CrowdFormationDetector(min_people=3, cluster_radius_ratio=0.25)
        people = [
            FakePerson(1, (10, 10, 30, 70)),
            FakePerson(2, (30, 10, 50, 70)),
            FakePerson(3, (50, 10, 70, 70)),
        ]

        assessments = detector.update(people, timestamp=4.0, frame_shape=(100, 100, 3))

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].detector, "crowd_formation")
        self.assertEqual(assessments[0].extra["people_count"], 3)

    def test_masked_entry_detector_emits_entry_check_once(self) -> None:
        detector = MaskedEntryCandidateDetector()
        inside = [FakeZoneState(4, (0, 0, 20, 80), zones=["front_entrance"])]

        first = detector.update(inside, timestamp=1.0)
        second = detector.update(inside, timestamp=2.0)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].detector, "masked_entry")
        self.assertTrue(first[0].extra["requires_visual_verification"])
        self.assertEqual(second, [])

    def test_situational_assessments_map_to_raw_events(self) -> None:
        detector = PersonDownDetector()
        assessments = detector.update(
            [FakePerson(9, (10, 60, 90, 100))],
            timestamp=2.0,
            frame_shape=(120, 120, 3),
        )

        events = situational_to_events(assessments)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].detector, "person_down")
        self.assertEqual(events[0].person_id, 9)


if __name__ == "__main__":
    unittest.main()
