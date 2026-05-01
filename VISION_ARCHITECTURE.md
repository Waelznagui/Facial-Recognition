# Vision Module — Architecture Brief
> This file is the source of truth for GitHub Copilot and any AI assistant
> working on the `vision/` package. Read it before touching any file in this folder.

---

## 1. What this module does

The `vision/` package is the perceptual brain of a study-assistant robot.
Every video frame goes through a pipeline that answers one question:
**"What is the user doing right now?"**

The answer is packaged into a shared state dict and emitted as typed events
to the rest of the system (hardware, session manager, LLM context).

---

## 2. File map and single responsibility

```
vision/
├── camera_capture.py        # SOURCE   — yields raw frames
├── vision_loop.py           # ORCHESTRATOR — calls all detectors per frame
├── face_detector.py         # DETECTOR — is a face present? where?
├── face_recognizer.py       # DETECTOR — who is it? (FaceNet profile match)
├── gaze_tracker.py          # DETECTOR — where is the user looking?
├── fatigue_detector.py      # DETECTOR — how drowsy is the user?
├── distraction_detector.py  # DETECTOR — is the user distracted?
├── vision_state_manager.py  # AGGREGATOR — merges all detector outputs
└── event_emitter.py         # PUBLISHER — fires typed events to subscribers
```

Each file owns exactly one concern. No file should import from a file
that is downstream of it in the pipeline.

---

## 3. Data flow (top to bottom)

```
camera_capture.py
      │  yields: np.ndarray (BGR frame)
      ▼
vision_loop.py
      │  calls each detector with the current frame + shared state
      │  collects their return dicts
      ▼
  ┌───────────────────────────────────────────────┐
  │  Detectors (run in this order every frame)    │
  │                                               │
  │  1. face_detector.py      → face_result       │
  │  2. face_recognizer.py    → identity_result   │  ← needs face_result
  │  3. gaze_tracker.py       → gaze_result       │  ← needs face_result
  │  4. fatigue_detector.py   → fatigue_result    │  ← needs face_result
  │  5. distraction_detector  → distraction_result│  ← needs all above
  └───────────────────────────────────────────────┘
      │  all result dicts passed together
      ▼
vision_state_manager.py
      │  merges dicts → single shared_state dict
      │  applies threshold + debounce logic
      │  decides final user_state label
      ▼
event_emitter.py
      │  publishes typed events
      ▼
  ┌────────────────────┬──────────────────┬────────────────────┐
  │ hardware_controller│session_controller│ ai_context_injector│
  │ motors·OLED·RGB·TTS│ pause timer · log│ state → LLM prompt │
  └────────────────────┴──────────────────┴────────────────────┘
```

---

## 4. Detector contract (every detector must follow this)

Every detector is a **class** with a single public method:

```python
class XxxDetector:
    def process(self, frame: np.ndarray, context: dict) -> dict:
        """
        Parameters
        ----------
        frame   : current BGR frame from camera_capture
        context : shared state dict built so far this tick
                  (contains outputs of all detectors that ran before this one)

        Returns
        -------
        dict    : flat dict of this detector's outputs.
                  Keys must be namespaced, e.g. "face_present", "gaze_yaw",
                  "fatigue_score" — never generic keys like "score" or "result".
        """
```

Rules:
- **Never block.** No sleep, no heavy I/O inside `process()`.
- **Never store frame references** across calls — only store derived metrics.
- **Return an empty dict `{}`** when the detector cannot run (e.g. no face found upstream).
- **Do not call other detectors** from inside `process()` — that is `vision_loop.py`'s job.

---

## 5. Shared state dict — key namespace

`vision_state_manager.py` merges all detector dicts into one flat dict.
To avoid collisions, every key is prefixed with its detector's name:

| Prefix         | Detector             | Example keys                                      |
|----------------|----------------------|---------------------------------------------------|
| `face_`        | face_detector        | `face_present`, `face_bbox`, `face_confidence`    |
| `identity_`    | face_recognizer      | `identity_name`, `identity_confidence`            |
| `gaze_`        | gaze_tracker         | `gaze_yaw`, `gaze_pitch`, `gaze_off_screen`       |
| `fatigue_`     | fatigue_detector     | `fatigue_score`, `fatigue_level`, `fatigue_perclos` |
| `distraction_` | distraction_detector | `distraction_type`, `distraction_score`, `phone_detected` |
| `state_`       | vision_state_manager | `state_user_state`, `state_confidence`, `state_since` |

---

## 6. Execution order and dependencies

```
face_detector       → no dependencies (runs on raw frame only)
face_recognizer     → requires: face_present == True
gaze_tracker        → requires: face_present == True
fatigue_detector    → requires: face_present == True
distraction_detector→ requires: face_present (for absence),
                               gaze_yaw / gaze_pitch (for gaze-off),
                               runs YOLOv8 independently for phone detection
vision_state_manager→ requires: all detector outputs
event_emitter       → requires: state_user_state from vision_state_manager
```

`vision_loop.py` enforces this order. If `face_present == False`, it
short-circuits: skips recognizer, gaze, fatigue, and passes `{}` for those
to `vision_state_manager` which handles the absent-user state.

---

## 7. Event types (event_emitter.py)

Events are typed dataclasses. The emitter publishes to a subscriber list.

```python
@dataclass
class VisionEvent:
    event_type: str          # "DISTRACTED" | "FATIGUED" | "FOCUSED" | "ABSENT" | "IDENTITY_FOUND"
    user_state: str          # current overall state label
    payload:    dict         # relevant metrics from shared_state
    timestamp:  float        # time.time()
    duration:   float        # seconds since this state began
```

Subscribers (hardware_controller, session_controller, ai_context_injector)
register a callback: `emitter.subscribe(callback_fn)`.
They must **not** call back into the vision pipeline — one-way flow only.

---

## 8. Models on disk

```
models/
├── facenet/          ← face_recognizer.py  (FaceNet weights)
├── yolov8_phone.pt   ← distraction_detector.py (phone detection)
└── (MediaPipe face_mesh is downloaded automatically by mediapipe)
```

Model paths are read from `config.py` — never hardcoded inside detector files.

---

## 9. What each file must NOT do

| File                    | Must NOT                                              |
|-------------------------|-------------------------------------------------------|
| `camera_capture.py`     | Process frames, import detectors                      |
| `vision_loop.py`        | Contain detection logic, emit events directly         |
| Any detector file       | Import another detector, emit events, write to DB     |
| `vision_state_manager`  | Run model inference, access camera                    |
| `event_emitter.py`      | Know what the subscribers will do with the event      |

---

## 10. Quick start for Copilot

When asked to implement or modify a file in `vision/`:

1. Check the **single responsibility** for that file (section 2).
2. Check its **dependencies** (section 6) — only use keys that upstream detectors provide.
3. Return a **namespaced dict** (section 5) — never mutate the context dict directly.
4. Follow the **detector contract** (section 4) — class with `process(frame, context)`.
5. If you need a model path, read it from `config.py`, not hardcoded.
