from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_action_model import VideoActionPrediction
from video_action_runtime import VideoActionRuntime


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def predict_frames(self, frames, *, top_k: int):
        self.calls.append([int(frame[0, 0, 0]) for frame in frames])
        return [VideoActionPrediction(label="punching person (boxing)", confidence=0.16, rank=1)]


class VideoActionRuntimeTests(unittest.TestCase):
    def test_analyze_event_uses_recent_window_centered_on_trigger_frame(self) -> None:
        model = FakeModel()
        runtime = VideoActionRuntime(
            model=model,
            backend="videomae",
            model_name="fake",
            fps=10.0,
            window_seconds=2.0,
            frame_count=5,
            top_k=3,
            cooldown_seconds=0.0,
        )
        for idx in range(30):
            runtime.add_frame(np.full((2, 2, 3), idx, dtype=np.uint8), frame_index=idx)

        events = runtime.analyze_event(center_frame_index=20, timestamp=2.0)

        self.assertEqual(model.calls, [[10, 14, 19, 24, 29]])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].detector, "video_action")
        self.assertEqual(events[0].extra["sampled_frame_indices"], [10, 14, 19, 24, 29])

    def test_analyze_event_respects_cooldown(self) -> None:
        model = FakeModel()
        runtime = VideoActionRuntime(
            model=model,
            backend="videomae",
            model_name="fake",
            fps=10.0,
            window_seconds=2.0,
            frame_count=5,
            top_k=3,
            cooldown_seconds=5.0,
        )
        for idx in range(30):
            runtime.add_frame(np.full((2, 2, 3), idx, dtype=np.uint8), frame_index=idx)

        first = runtime.analyze_event(center_frame_index=20, timestamp=2.0)
        second = runtime.analyze_event(center_frame_index=21, timestamp=3.0)

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(len(model.calls), 1)

    def test_analyze_event_exposes_sampled_bgr_frames_for_gate(self) -> None:
        model = FakeModel()
        runtime = VideoActionRuntime(
            model=model,
            backend="videomae",
            model_name="fake",
            fps=10.0,
            window_seconds=2.0,
            frame_count=5,
            top_k=3,
            cooldown_seconds=0.0,
        )
        for idx in range(30):
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            frame[:, :, 0] = idx
            frame[:, :, 1] = idx + 1
            frame[:, :, 2] = idx + 2
            runtime.add_frame(frame, frame_index=idx)

        runtime.analyze_event(center_frame_index=20, timestamp=2.0)

        gate_frames = runtime.last_gate_frames()
        self.assertEqual(runtime.last_sampled_frame_indices, [10, 14, 19, 24, 29])
        self.assertEqual([int(frame[0, 0, 0]) for frame in gate_frames], [10, 14, 19, 24, 29])
        self.assertEqual([int(frame[0, 0, 2]) for frame in gate_frames], [12, 16, 21, 26, 31])

    def test_zero_cooldown_is_normalized_to_window_duration(self) -> None:
        model = FakeModel()
        runtime = VideoActionRuntime(
            model=model,
            backend="videomae",
            model_name="fake",
            fps=10.0,
            window_seconds=2.0,
            frame_count=5,
            top_k=3,
            cooldown_seconds=0.0,
        )
        for idx in range(30):
            runtime.add_frame(np.full((2, 2, 3), idx, dtype=np.uint8), frame_index=idx)

        first = runtime.analyze_event(center_frame_index=20, timestamp=2.0)
        second = runtime.analyze_event(center_frame_index=21, timestamp=2.5)

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(len(model.calls), 1)

    def test_last_gate_frames_can_be_capped_for_small_vlms(self) -> None:
        model = FakeModel()
        runtime = VideoActionRuntime(
            model=model,
            backend="videomae",
            model_name="fake",
            fps=10.0,
            window_seconds=2.0,
            frame_count=5,
            top_k=3,
            cooldown_seconds=0.0,
        )
        for idx in range(30):
            runtime.add_frame(np.full((2, 2, 3), idx, dtype=np.uint8), frame_index=idx)

        runtime.analyze_event(center_frame_index=20, timestamp=2.0)

        gate_frames = runtime.last_gate_frames(max_count=3)
        self.assertEqual([int(frame[0, 0, 0]) for frame in gate_frames], [10, 19, 29])


if __name__ == "__main__":
    unittest.main()
