import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

# ─────────────────────────────────────────────
#  LANDMARK INDICES & 3D MODEL
# ─────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [78,  81,  311, 308, 402, 178]

# 6 anchor points for solvePnP (generic 3-D face model in mm)
FACE_3D = np.array([
    [  0.0,    0.0,    0.0],   # Nose tip          → lm 4
    [  0.0, -330.0,  -65.0],   # Chin              → lm 152
    [-225.0,  170.0, -135.0],  # Left eye corner   → lm 263
    [ 225.0,  170.0, -135.0],  # Right eye corner  → lm 33
    [-150.0, -150.0, -125.0],  # Left mouth corner → lm 287
    [ 150.0, -150.0, -125.0],  # Right mouth corner→ lm 57
], dtype=np.float64)
POSE_IDS = [4, 152, 263, 33, 287, 57]

class FatigueDetector:
    """
    Fatigue Detector conforming to the VISION_ARCHITECTURE.md contract.
    """

    def __init__(self,
                 calibration_frames: int = 120,   # ~4 s at 30 fps
                 perclos_window_sec: int = 60,
                 score_smooth_sec:   int = 3):

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # MediaPipe Face Mesh (Tasks API for Python 3.12 compatibility)
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.face_mesh = vision.FaceLandmarker.create_from_options(options)

        # ── Calibration ──────────────────────────────────────
        self.calibration_frames   = calibration_frames
        self._cal_ear:   list     = []
        self._cal_pitch: list     = []
        self.is_calibrated        = False
        self.EAR_THRESHOLD        = 0.20   # overwritten after calibration
        self.neutral_pitch        = 0.0    # overwritten after calibration
        self.MAR_THRESHOLD        = 0.60   # fixed (yawn shape is universal)
        self.PITCH_NOD_OFFSET     = -15.0  # degrees BELOW neutral pitch = nodding

        # ── PERCLOS ───────────────────────────────────────────
        self._perclos_win: deque  = deque(maxlen=9000)  # overwritten dynamically
        self._fps_win:     deque  = deque(maxlen=30)
        self._last_t:      float  = 0.0
        self._measured_fps: float = 30.0
        self._perclos_window_sec  = perclos_window_sec

        # ── Blink tracking ────────────────────────────────────
        self._eye_was_open: bool  = True
        self._blink_times:  deque = deque()   # timestamp of each blink

        # ── Yawn tracking ────────────────────────────────────
        self._yawn_frames:  int   = 0
        self._is_yawning:   bool  = False
        self._yawn_times:   deque = deque()
        self.YAWN_FRAME_THRESHOLD = 35

        # ── Nod tracking ─────────────────────────────────────
        self._nod_frames:   int   = 0
        self._is_nodding:   bool  = False
        self._nod_times:    deque = deque()
        self.NOD_FRAME_THRESHOLD  = 25

        # ── Score smoothing ───────────────────────────────────
        self._score_hist:   deque = deque(maxlen=int(30 * score_smooth_sec))
        self._score_smooth_sec    = score_smooth_sec

    @staticmethod
    def _aspect_ratio(landmarks, ids, w, h) -> float:
        pts = np.array([[landmarks.landmark[i].x * w,
                         landmarks.landmark[i].y * h] for i in ids])
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        hz = np.linalg.norm(pts[0] - pts[3])
        return float((v1 + v2) / (2.0 * hz + 1e-6))

    @staticmethod
    def _head_pose(landmarks, w, h):
        """solvePnP → pitch, yaw, roll in degrees."""
        img_pts = np.array(
            [[landmarks.landmark[i].x * w,
              landmarks.landmark[i].y * h] for i in POSE_IDS],
            dtype=np.float64
        )
        focal = w
        cam_matrix = np.array([[focal, 0, w / 2],
                                [0, focal, h / 2],
                                [0,     0,     1]], dtype=np.float64)
        dist = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(FACE_3D, img_pts, cam_matrix, dist,
                                    flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0, 0.0, 0.0
        R, _ = cv2.Rodrigues(rvec)
        sy  = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw   = math.degrees(math.atan2( R[2, 1], R[2, 2]))
        roll  = math.degrees(math.atan2( R[1, 0], R[0, 0]))
        return pitch, yaw, roll

    @staticmethod
    def _purge_old(dq: deque, now: float, max_age: float = 60.0):
        while dq and now - dq[0] > max_age:
            dq.popleft()

    def _run_calibration(self, ear: float, pitch: float):
        self._cal_ear.append(ear)
        self._cal_pitch.append(pitch)
        if len(self._cal_ear) >= self.calibration_frames:
            baseline_ear        = float(np.mean(self._cal_ear))
            self.EAR_THRESHOLD  = baseline_ear * 0.75   # 75 % of open-eye baseline
            self.neutral_pitch  = float(np.mean(self._cal_pitch))
            self.is_calibrated  = True

    def process(self, frame: np.ndarray, context: dict) -> dict:
            """
            Complies with the VISION_ARCHITECTURE.md contract.
            Extracts fatigue metrics and returns a namespaced dictionary.
            """
            if context.get("face_present") is False:
                return {}

            now = time.time()
            
            # FIX 1: Calculate dt BEFORE returning due to missing landmarks.
            # This prevents massive time spikes from corrupting the FPS calculation.
            dt = now - self._last_t if self._last_t else 0.0
            self._last_t = now

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = self.face_mesh.detect(mp_image)

            if not detection_result.face_landmarks:
                return {}

            class MockLandmarks:
                def __init__(self, lm):
                    self.landmark = lm

            landmarks = MockLandmarks(detection_result.face_landmarks[0])

            # FIX 2: Only dynamically resize if the framerate difference is realistic.
            # Ignore massive dt spikes (e.g., face was missing for 2 seconds)
            if 0 < dt < 0.5:
                self._fps_win.append(1.0 / dt)
                self._measured_fps = float(np.mean(self._fps_win))
                
                # Use max(1, ...) to prevent deques from collapsing to size 0
                new_maxlen = max(1, int(self._measured_fps * self._perclos_window_sec))
                if new_maxlen != self._perclos_win.maxlen:
                    self._perclos_win = deque(self._perclos_win, maxlen=new_maxlen)
                    
                new_score_maxlen = max(1, int(self._measured_fps * self._score_smooth_sec))
                if new_score_maxlen != self._score_hist.maxlen:
                    self._score_hist = deque(self._score_hist, maxlen=new_score_maxlen)

            # Raw metrics
            left_ear  = self._aspect_ratio(landmarks, LEFT_EYE,  w, h)
            right_ear = self._aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0
            mar       = self._aspect_ratio(landmarks, MOUTH,     w, h)
            pitch, yaw, roll = self._head_pose(landmarks, w, h)

            # Calibration phase
            if not self.is_calibrated:
                self._run_calibration(avg_ear, pitch)
                return {
                    "fatigue_is_calibrated": False,
                    "fatigue_calibration_progress": round(len(self._cal_ear) / self.calibration_frames, 2),
                    "fatigue_avg_ear": round(avg_ear, 3),
                    "fatigue_mar": round(mar, 3),
                    "fatigue_pitch": round(pitch, 1),
                }

            relative_pitch = pitch - self.neutral_pitch

            # FIX 3: Guard EAR against extreme head rotations.
            # When looking away, the 2D eye aspect ratio shrinks artificially. 
            # We flag the head as turned and prevent false "eye closed" triggers.
            head_turned = abs(yaw) > 30 or abs(relative_pitch) > 30
            eyes_closed = (avg_ear < self.EAR_THRESHOLD) and not head_turned

            # PERCLOS (Percentage of Eye Closure)
            self._perclos_win.append(1 if eyes_closed else 0)
            perclos = sum(self._perclos_win) / max(len(self._perclos_win), 1)

            # Blink rate
            just_blinked = (self._eye_was_open and eyes_closed)
            if not self._eye_was_open and not eyes_closed:
                self._blink_times.append(now)
            self._eye_was_open = not eyes_closed
            self._purge_old(self._blink_times, now, 60)
            blink_rate = len(self._blink_times)
            blink_too_slow = (blink_rate < 8 and len(self._perclos_win) >= int(self._measured_fps * 15))

            # Yawn rate
            if mar > self.MAR_THRESHOLD:
                self._yawn_frames += 1
            else:
                self._yawn_frames = max(0, self._yawn_frames - 1)

            if self._yawn_frames > self.YAWN_FRAME_THRESHOLD and not self._is_yawning:
                self._is_yawning = True
                self._yawn_times.append(now)
            if self._yawn_frames == 0:
                self._is_yawning = False
            self._purge_old(self._yawn_times, now, 60)
            yawn_rate = len(self._yawn_times)

            # Nod rate
            if relative_pitch < self.PITCH_NOD_OFFSET:
                self._nod_frames += 1
            else:
                self._nod_frames = max(0, self._nod_frames - 1)
                
            if self._nod_frames > self.NOD_FRAME_THRESHOLD and not self._is_nodding:
                self._is_nodding = True
                self._nod_times.append(now)
            if self._nod_frames == 0:
                self._is_nodding = False
            self._purge_old(self._nod_times, now, 60)
            nod_rate = len(self._nod_times)

            # Composite Fatigue Score [0.0 - 1.0]
            n_perclos = min(1.0, perclos / 0.30)            # 30 % closure → max
            n_yawns   = min(1.0, yawn_rate / 3.0)           # 3 yawns / min → max
            n_nods    = min(1.0, nod_rate  / 3.0)           # 3 nods  / min → max
            n_blink   = 1.0 if blink_too_slow else 0.0

            raw_score = (n_perclos * 0.45 + n_yawns * 0.25 + n_nods * 0.20 + n_blink * 0.10)
            self._score_hist.append(raw_score)
            fatigue_score = float(np.mean(self._score_hist))

            # Set Fatigue Level
            if fatigue_score > 0.70:
                level = "Critical"
            elif fatigue_score > 0.40:
                level = "Mild"
            else:
                level = "Awake"

            # Return Namespaced Dict per Architectural Contract
            return {
                "fatigue_is_calibrated": True,
                "fatigue_avg_ear":       round(avg_ear, 3),
                "fatigue_mar":           round(mar, 3),
                "fatigue_pitch":         round(pitch, 1),
                "fatigue_yaw":           round(yaw, 1),
                "fatigue_roll":          round(roll, 1),
                "fatigue_relative_pitch":round(relative_pitch, 1),
                "fatigue_perclos":       round(perclos, 4),
                "fatigue_blink_rate":    blink_rate,
                "fatigue_blink_too_slow":blink_too_slow,
                "fatigue_yawn_rate":     yawn_rate,
                "fatigue_nod_rate":      nod_rate,
                "fatigue_score":         round(fatigue_score, 4),
                "fatigue_level":         level,
                "fatigue_measured_fps":  round(self._measured_fps, 1),
            }
fatigue_detector = FatigueDetector()