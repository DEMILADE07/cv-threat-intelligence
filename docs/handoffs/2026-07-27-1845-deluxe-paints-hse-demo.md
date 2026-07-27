# Deluxe Paints HSE Demo: Fire/Smoke, Running, Crowd Formation

> **Written:** 2026-07-27 18:45 WAT (Monday)
> **Author:** Codex with Demi
> **Branch:** `test-ayo-main`
> **Base commit:** `555498a chore(configs): update site demo configs to use repo local test_clips and anomaly video paths`

## Why This Exists

We have a Deluxe Paints Nigeria manufacturing/HSE demo today. Training or
finetuning a new model was rejected as too risky for the short timeframe. The
goal was to add a credible safety-oriented demo layer that fits Ayo's
productionized path:

```text
cvti.serving.pipeline
  -> per-camera detectors
  -> RawEvent
  -> CustomizationEngine
  -> local Ollama/Gemma VLM gate
  -> events.db / operator UI
```

The new HSE layer is intentionally a **candidate generator**, not the final
judge. It creates cheap local signals; the existing VLM gate confirms/rejects
with manufacturing scene context.

## What Was Added

### 1. Lightweight HSE Detectors

New file:

```text
cvti/detector/situational.py
```

Classes:

- `FireSmokeCandidateDetector`
  - Uses HSV masks for flame-colored regions and smoke-like regions.
  - Has temporal persistence (`min_frames`) so a single weird frame does not fire.
  - Includes a grey-wall guard: smoke ratio must not be almost the whole frame,
    because plain grey walls/warehouses caused a false fire/smoke trigger.

- `RunningPanicDetector`
  - Tracks bbox center movement per `track_id`.
  - Converts pixel speed into frame-diagonal-normalized speed.
  - Fires after sustained fast motion for several frames.

- `CrowdFormationDetector`
  - Takes tracked people bboxes.
  - Finds tight clusters by center distance.
  - Fires only when enough people cluster for enough frames.

Tests:

```text
tests/test_situational_hse.py
```

The tests cover:

- sustained fast movement fires running
- slow movement stays quiet
- persistent tight cluster fires crowd formation
- spread-out people stay quiet
- flame-colored frame fires fire/smoke
- dark frame and plain grey wall stay quiet

### 2. Productionized Serving Integration

Modified file:

```text
cvti/serving/camera.py
```

New per-camera config toggles:

```json
"fire_smoke": true,
"running": true,
"crowd_formation": true
```

Optional per-camera tuning knobs:

```json
"running_min_speed_ratio": 0.07,
"running_min_frames": 3,
"crowd_min_people": 3,
"crowd_min_frames": 2,
"crowd_max_cluster_ratio": 0.32,
"fire_min_frames": 3,
"fire_min_hot_area_ratio": 0.012
```

The detectors emit shared events:

```text
RawEvent(detector="fire", level="critical", title="POSSIBLE FIRE OR SMOKE")
RawEvent(detector="running", level="high", title="PANIC RUNNING DETECTED")
RawEvent(detector="crowd_formation", level="medium", title="UNSAFE CROWD FORMATION")
```

These flow through the existing `CustomizationEngine`, `AlertQueue`,
`GatePool`, `VerificationGate`, `AlertSink`, `events.db`, and the operator UI.

### 3. HSE Rules and Demo Configs

New config:

```text
configs/manufacturing_hse_v1.json
```

Rules:

```text
panic_running -> detector "running" -> high
unsafe_crowd_formation -> detector "crowd_formation" -> medium
```

Fire/smoke uses the existing always-on critical baseline:

```text
baseline_fire_smoke -> detector "fire" -> critical
```

New site config:

```text
configs/deluxe_paints_demo.json
```

Current demo cameras:

```text
production_floor_fire_watch
  source: data/hse_demo/fire_videos.1406/pos/posVideo1.868.mp4
  detector: fire_smoke

factory_aisle_panic_watch
  source: data/hse_demo/Crowd-Activity-All.mp4
  detector: running
  tuned at running_min_speed_ratio=0.07, running_min_frames=3

assembly_area_crowd_watch
  source: data/anomaly/119.mp4
  detector: crowd_formation
  tuned at crowd_min_people=3, crowd_min_frames=2, crowd_max_cluster_ratio=0.32
```

### 4. VLM Gate Questions

Modified file:

```text
cvti/verification/gate.py
```

Added HSE-specific questions so Gemma is not asked a vague generic threat
question:

```text
baseline_fire_smoke / fire_smoke:
  Does this frame show visible fire, flames, smoke, or hazardous haze in a manufacturing_floor?

panic_running:
  Does this brief sequence show a person running or moving with panic/urgency in a manufacturing_floor?

unsafe_crowd_formation:
  Does this frame show an unsafe crowd or tight group formation in a manufacturing_floor?
```

### 5. AVI to MP4 Converter

New script:

```text
scripts/convert_avi_to_mp4.py
```

Reason: the downloaded HSE datasets are `.avi`, and QuickTime does not play many
of those codecs. The script uses OpenCV to re-encode to `.mp4` so Demi can
preview clips in Finder/QuickTime and the demo config can point at MP4s.

Usage:

```bash
./.venv/bin/python scripts/convert_avi_to_mp4.py data/hse_demo/fire_videos.1406 --overwrite
./.venv/bin/python scripts/convert_avi_to_mp4.py data/hse_demo/smoke_videos.1407 --overwrite
./.venv/bin/python scripts/convert_avi_to_mp4.py data/hse_demo/Crowd-Activity-All.avi --overwrite
```

## HSE Data Assets

Local data folder:

```text
data/hse_demo/
```

Current size:

```text
~2.5G
```

Contents:

```text
Crowd-Activity-All.avi
Crowd-Activity-All.mp4
fire_videos.1406.zip
fire_videos.1406/
smoke_videos.1407.zip
smoke_videos.1407/
```

Original source links used:

```text
Fire/smoke FIRESENSE Zenodo:
https://zenodo.org/records/836749

UMN Crowd Activity AVI:
http://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi
```

Conversion status:

```text
49 original extracted AVI files
49 converted MP4 files
```

Important: media files are ignored by `.gitignore`:

```text
*.zip
data/**/*.mp4
data/**/*.avi
data/anomaly/
```

Do **not** force-push the full `data/hse_demo` folder to GitHub. It is too large
and includes files over GitHub's normal 100MB file limit:

```text
data/hse_demo/fire_videos.1406.zip                  ~594M
data/hse_demo/smoke_videos.1407.zip                 ~188M
data/hse_demo/fire_videos.1406/neg/negsVideo15...   ~319M
```

Recommended sharing path:

1. Share the source links above with Ayo.
2. Or copy `data/hse_demo` over AirDrop/Drive/USB.
3. Or upload the dataset folder to Drive/S3 and keep Git as code-only.
4. Only use Git LFS if the team explicitly decides to store demo assets in the repo.

## Verification Done

Focused tests:

```bash
./.venv/bin/python -m pytest tests/test_situational_hse.py tests/test_serving.py -q
```

Result:

```text
19 passed
```

Mock productionized demo:

```bash
MPLCONFIGDIR=/private/tmp ./.venv/bin/python -m cvti.serving.pipeline \
  --site-config configs/deluxe_paints_demo.json \
  --gate-provider mock \
  --notify console \
  --output-dir runs/deluxe_paints_mock \
  --target-fps 4 \
  --imgsz 512 \
  --seconds 20 \
  --gate-drain 10
```

Expected behavior:

```text
baseline_fire_smoke alert from production_floor_fire_watch
panic_running alert from factory_aisle_panic_watch
unsafe_crowd_formation alert from assembly_area_crowd_watch
```

Demi's last mock run before the full VLM test produced 2 alerts in 20s:

```text
baseline_fire_smoke
unsafe_crowd_formation
```

That is acceptable for a short mock run because the UMN running clip has the
panic burst later. Use 60-120s for the full demo if you want running to appear.

## Full End-to-End Local Ollama Test

Terminal 1:

```bash
ollama serve
```

If it says `address already in use`, Ollama is already running. Confirm with:

```bash
curl http://localhost:11434/api/version
ollama list
```

Terminal 2, engine:

```bash
cd "/Users/macbook/Desktop/CV Threat Intelligence/cv-threat-intelligence"
source .venv/bin/activate

OLLAMA_API_KEY=ollama MPLCONFIGDIR=/private/tmp ./.venv/bin/python -m cvti.serving.pipeline \
  --site-config configs/deluxe_paints_demo.json \
  --gate-provider ollama \
  --gate-model gemma3:4b \
  --notify console \
  --output-dir runs/deluxe_paints_live \
  --target-fps 4 \
  --imgsz 512 \
  --seconds 120 \
  --gate-drain 180
```

Terminal 3, operator UI:

```bash
cd "/Users/macbook/Desktop/CV Threat Intelligence/cv-threat-intelligence"
source .venv/bin/activate

./.venv/bin/python -m cvti.app.shell \
  --site-config configs/deluxe_paints_demo.json \
  --db runs/deluxe_paints_live/events.db
```

What the UI shows today:

```text
Live tab:
  video wall + alerting tiles/banners

Alerts tab:
  saved confirmed alerts and evidence frames/event playback

It does NOT currently draw YOLO boxes live over every frame in the productionized
web UI. That would be a separate UI overlay task.
```

## Integration Notes for Ayo's Agent

1. Pull/merge the code files first, without media:

```text
cvti/detector/situational.py
cvti/serving/camera.py
cvti/verification/gate.py
configs/manufacturing_hse_v1.json
configs/deluxe_paints_demo.json
scripts/convert_avi_to_mp4.py
tests/test_situational_hse.py
```

2. Keep the HSE detectors as **candidate generators**. Do not let them bypass
the VLM gate except for mock/development testing.

3. The productionized UI should surface these rule names cleanly:

```text
baseline_fire_smoke
panic_running
unsafe_crowd_formation
```

4. If adding live video overlays, use the detector `extra` payloads:

```text
fire: hot_area_ratio, smoke_area_ratio
running: track_id, bbox, speed_ratio, fast_frames
crowd_formation: bbox, people_count, track_ids
```

5. For a cleaner demo, keep weapons disabled in this HSE config. This demo is
about manufacturing safety, not retail robbery.

6. If the running clip seems quiet in a short 20s run, extend the engine to
60-120s. The UMN clip has the panic/running segment later.

## Known Caveats

- Fire/smoke detector is color/texture heuristic, not a trained fire model. It
is deliberately verified by Gemma before becoming a real alert.
- Smoke is difficult with plain grey/white walls; the heuristic includes a
grey-wall guard but should not be considered production fire detection.
- Crowd formation is spatial, not semantic. It finds clusters, not intent.
- Running depends on tracking stability and frame rate. Thresholds are tuned for
the demo clips, not universal deployment.
- The productionized UI currently shows video wall + alert evidence, not
live-drawn bounding boxes on every frame.

## Recommended Next Step

For today's demo, Ayo should integrate the code-side changes and keep media
outside Git. If he needs the exact assets, copy `data/hse_demo` directly or
download from the two links and run `scripts/convert_avi_to_mp4.py`.

