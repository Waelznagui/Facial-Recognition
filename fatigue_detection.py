import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

# ─────────────────────────────────────────────
#  LANDMARK INDICES
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


# ─────────────────────────────────────────────
#  FATIGUE DETECTOR CLASS
# ─────────────────────────────────────────────
class FatigueDetector:
    """
    Pipeline-ready fatigue detector.
    Call process(landmarks, frame_w, frame_h, timestamp) each frame.
    Returns a dict with all metrics + a fatigue_score in [0, 1].
    Call result["is_calibrated"] to know if calibration is done.
    """

    def __init__(self,
                 calibration_frames: int = 120,   # ~4 s at 30 fps
                 perclos_window_sec: int = 60,
                 score_smooth_sec:   int = 3):

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
        # filled dynamically once fps is known
        self._perclos_win: deque  = deque(maxlen=9000)  # overwritten
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

    # ─────────────── helpers ──────────────────────────────────

    @staticmethod
    def _aspect_ratio(landmarks, ids, w, h) -> float:
        pts = np.array([[landmarks.landmark[i].x * w,
                         landmarks.landmark[i].y * h] for i in ids])
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        hz = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    @staticmethod
    def _head_pose(landmarks, w, h):
        """solvePnP → pitch, yaw, roll in degrees. Much more reliable than Z-delta."""
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

    # ─────────────── calibration ──────────────────────────────

    def _run_calibration(self, ear: float, pitch: float):
        self._cal_ear.append(ear)
        self._cal_pitch.append(pitch)
        if len(self._cal_ear) >= self.calibration_frames:
            baseline_ear        = float(np.mean(self._cal_ear))
            self.EAR_THRESHOLD  = baseline_ear * 0.75   # 75 % of open-eye baseline
            self.neutral_pitch  = float(np.mean(self._cal_pitch))
            self.is_calibrated  = True

    # ─────────────── main entry point ─────────────────────────

    def process(self, landmarks, w: int, h: int, now: float) -> dict:
        """
        Parameters
        ----------
        landmarks : mediapipe face landmark object
        w, h      : frame width / height in pixels
        now       : current timestamp (time.time())

        Returns
        -------
        dict with keys:
          is_calibrated, calibration_progress,
          avg_ear, mar, pitch, yaw, roll, relative_pitch,
          perclos, blink_rate, yawn_rate, nod_rate,
          fatigue_score  (0-1),  fatigue_level ("Awake" / "Mild" / "Critical")
        """

        # ── Measure real FPS ─────────────────────────────────
        if self._last_t:
            dt = now - self._last_t
            if dt > 0:
                self._fps_win.append(1.0 / dt)
                self._measured_fps = float(np.mean(self._fps_win))
                new_maxlen = int(self._measured_fps * self._perclos_window_sec)
                if new_maxlen != self._perclos_win.maxlen:
                    self._perclos_win = deque(self._perclos_win, maxlen=new_maxlen)
                new_score_maxlen = int(self._measured_fps * self._score_smooth_sec)
                if new_score_maxlen != self._score_hist.maxlen:
                    self._score_hist = deque(self._score_hist, maxlen=new_score_maxlen)
        self._last_t = now

        # ── Raw metrics ──────────────────────────────────────
        left_ear  = self._aspect_ratio(landmarks, LEFT_EYE,  w, h)
        right_ear = self._aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0
        mar       = self._aspect_ratio(landmarks, MOUTH,     w, h)
        pitch, yaw, roll = self._head_pose(landmarks, w, h)

        # ── Calibration phase ────────────────────────────────
        if not self.is_calibrated:
            self._run_calibration(avg_ear, pitch)
            return {
                "is_calibrated":        False,
                "calibration_progress": len(self._cal_ear) / self.calibration_frames,
                "avg_ear": avg_ear, "mar": mar,
                "pitch": pitch, "yaw": yaw, "roll": roll,
            }

        relative_pitch = pitch - self.neutral_pitch

        # ── PERCLOS ──────────────────────────────────────────
        eyes_closed = avg_ear < self.EAR_THRESHOLD
        self._perclos_win.append(1 if eyes_closed else 0)
        perclos = sum(self._perclos_win) / max(len(self._perclos_win), 1)

        # ── Blink rate  (open → close → open = 1 blink) ──────
        just_blinked = (self._eye_was_open and eyes_closed)  # leading edge
        if not self._eye_was_open and not eyes_closed:        # trailing edge = completed blink
            self._blink_times.append(now)
        self._eye_was_open = not eyes_closed
        self._purge_old(self._blink_times, now, 60)
        blink_rate = len(self._blink_times)   # blinks / min
        # Normal ≈ 12-20 bpm.  < 8 bpm = drowsy / staring.  > 25 bpm = dry / stressed.
        # Low blink rate is a fatigue signal.
        blink_too_slow = (blink_rate < 8
                          and len(self._perclos_win) >= int(self._measured_fps * 15))

        # ── Yawn rate ─────────────────────────────────────────
        self._yawn_frames = (self._yawn_frames + 1
                             if mar > self.MAR_THRESHOLD
                             else max(0, self._yawn_frames - 1))
        if self._yawn_frames > self.YAWN_FRAME_THRESHOLD and not self._is_yawning:
            self._is_yawning = True
            self._yawn_times.append(now)
        if self._yawn_frames == 0:
            self._is_yawning = False
        self._purge_old(self._yawn_times, now, 60)
        yawn_rate = len(self._yawn_times)

        # ── Nod rate ──────────────────────────────────────────
        self._nod_frames = (self._nod_frames + 1
                            if relative_pitch < self.PITCH_NOD_OFFSET
                            else max(0, self._nod_frames - 1))
        if self._nod_frames > self.NOD_FRAME_THRESHOLD and not self._is_nodding:
            self._is_nodding = True
            self._nod_times.append(now)
        if self._nod_frames == 0:
            self._is_nodding = False
        self._purge_old(self._nod_times, now, 60)
        nod_rate = len(self._nod_times)

        # ── Composite fatigue score ───────────────────────────
        # Normalise each signal to [0, 1]
        n_perclos = min(1.0, perclos / 0.30)            # 30 % closure → max
        n_yawns   = min(1.0, yawn_rate / 3.0)           # 3 yawns / min → max
        n_nods    = min(1.0, nod_rate  / 3.0)           # 3 nods  / min → max
        n_blink   = 1.0 if blink_too_slow else 0.0      # binary flag

        raw_score = (n_perclos * 0.45 +
                     n_yawns   * 0.25 +
                     n_nods    * 0.20 +
                     n_blink   * 0.10)

        # Smooth score over several seconds to remove flicker
        self._score_hist.append(raw_score)
        fatigue_score = float(np.mean(self._score_hist))

        # ── Fatigue level ─────────────────────────────────────
        if fatigue_score > 0.70:
            level = "Critical"
        elif fatigue_score > 0.40:
            level = "Mild"
        else:
            level = "Awake"

        return {
            "is_calibrated":    True,
            "avg_ear":          round(avg_ear, 3),
            "mar":              round(mar, 3),
            "pitch":            round(pitch, 1),
            "yaw":              round(yaw, 1),
            "roll":             round(roll, 1),
            "relative_pitch":   round(relative_pitch, 1),
            "perclos":          round(perclos, 4),
            "blink_rate":       blink_rate,
            "blink_too_slow":   blink_too_slow,
            "yawn_rate":        yawn_rate,
            "nod_rate":         nod_rate,
            "fatigue_score":    round(fatigue_score, 4),
            "fatigue_level":    level,
            "fps":              round(self._measured_fps, 1),
        }


# ─────────────────────────────────────────────
#  STANDALONE DEMO  (python fatigue_detector.py)
# ─────────────────────────────────────────────
def _draw_bar(frame, x, y, w, value, color, label):
    cv2.rectangle(frame, (x, y), (x + w, y + 16), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * value), y + 16), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + 16), (130, 130, 130), 1)
    cv2.putText(frame, label, (x + w + 8, y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = FatigueDetector(calibration_frames=120)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = face_mesh.process(rgb)
        now   = time.time()

        overlay = frame.copy()

        if res.multi_face_landmarks:
            for lm in res.multi_face_landmarks:
                result = detector.process(lm, w, h, now)

                # ── Calibration screen ─────────────────────────
                if not result["is_calibrated"]:
                    prog = result["calibration_progress"]
                    cv2.putText(frame, "Calibrating — look straight ahead",
                                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 200, 255), 2)
                    cv2.rectangle(frame, (20, 70), (20 + int(prog * 300), 90),
                                  (0, 200, 255), -1)
                    cv2.rectangle(frame, (20, 70), (320, 90), (130, 130, 130), 1)
                    continue

                # ── Colour by level ────────────────────────────
                lvl   = result["fatigue_level"]
                score = result["fatigue_score"]
                color = {
                    "Awake":    (0,  210,  80),
                    "Mild":     (0,  165, 255),
                    "Critical": (0,   0,  230),
                }[lvl]

                # ── Status banner ──────────────────────────────
                cv2.rectangle(overlay, (0, 0), (w, 40), color, -1)
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                cv2.putText(frame, lvl.upper(), (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # ── Score bar ──────────────────────────────────
                _draw_bar(frame, 12, 50, 260, score, color,
                          f"Score: {int(score*100)}%")

                # ── Metric readouts ────────────────────────────
                metrics = [
                    f"PERCLOS: {result['perclos']*100:.1f}%",
                    f"EAR: {result['avg_ear']:.3f}  (thr {detector.EAR_THRESHOLD:.3f})",
                    f"MAR: {result['mar']:.3f}",
                    f"Pitch: {result['relative_pitch']:+.1f}°  Yaw: {result['yaw']:+.1f}°",
                    f"Blink: {result['blink_rate']} bpm {'⚠ LOW' if result['blink_too_slow'] else ''}",
                    f"Yawns/min: {result['yawn_rate']}   Nods/min: {result['nod_rate']}",
                    f"FPS: {result['fps']}",
                ]
                for i, txt in enumerate(metrics):
                    cv2.putText(frame, txt, (12, 92 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (210, 210, 210), 1)

                # ── Landmark dots ──────────────────────────────
                for p in LEFT_EYE + RIGHT_EYE + MOUTH:
                    pt = lm.landmark[p]
                    cv2.circle(frame,
                               (int(pt.x * w), int(pt.y * h)),
                               1, (0, 255, 200), -1)

        else:
            cv2.putText(frame, "No face detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)

        cv2.imshow("Fatigue Detector", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()