import cv2
import threading
import time
import os
import numpy as np
from deepface import DeepFace

class FaceRecognizer:
    """DETECTOR -> who is it? (Uses DeepFace asynchronously)"""
    def __init__(self, db_path="faces_db", interval=1.0):
        self.db_path = db_path
        self.interval = interval     # How often to query DeepFace
        self._last_run = time.time()
        self._is_recognizing = False
        
        # Start with unknown state
        self._identity_name = "Unknown"
        self._identity_distance = 1.0

        # TENSORFLOW FIX: Load the models on the MAIN thread before backgrounding.
        # Background threads crash silently if TF builds the graph for the first time there.
        print("Initializing DeepFace Models on main thread (Please wait)...")
        try:
            # We don't save the result, just let it cache weights in memory
            DeepFace.build_model('ArcFace')
            # DeepFace.build_model('retinaface') # Removed to prevent Keras Threading Crash
            print("DeepFace Models Loaded!")
        except Exception as e:
            print(f"Warning: Could not pre-cache models: {e}")

    def _recognize_worker(self, frame_copy):
        """Asynchronous worker to avoid blocking the main vision loop."""
        try:
            results = DeepFace.find(
                img_path=frame_copy, 
                db_path=self.db_path, 
                model_name='ArcFace', 
                detector_backend='opencv', 
                enforce_detection=False,
                distance_metric='cosine',
                silent=True,
                anti_spoofing=False
            )
            
            if results and not results[0].empty:
                best_match = results[0].iloc[0]
                distance = best_match["distance"]
                
                # DeepFace calculates this automatically but 0.68 is optimal for ArcFace
                if distance <= 0.68:
                    identity_path = best_match['identity']
                    name = os.path.basename(os.path.dirname(identity_path))
                    if name == "faces_db" or name == "face_recognition": 
                        name = os.path.splitext(os.path.basename(identity_path))[0]
                        
                    print(f"!!! DEEPFACE FOUND YOU !!! Name: {name}, Distance: {distance}")
                    self._identity_name = name
                    self._identity_distance = distance
                else:
                    self._identity_name = "Unknown"
                    self._identity_distance = distance
            else:
                self._identity_name = "Unknown"
                self._identity_distance = 1.0
                
        except Exception as e:
            # Catch DeepFace errors instead of silencing them so we can debug!
            print(f"[DeepFace Thread Crash] {e}")
            pass
        finally:
            self._is_recognizing = False

    def process(self, frame: np.ndarray, context: dict) -> dict:
        """
        Polls the background AI thread for the latest known identity.
        If it's time to check again, spins up the worker.
        """
        # Contract: Do not start a NEW recognition if no face is found upstream.
        # But do NOT erase the current known identity just because of a 1-frame blink!
        if not context.get("face_present", False):
            return {
                "identity_name": self._identity_name,
                "identity_confidence_distance": round(self._identity_distance, 3),
                "identity_is_thinking": self._is_recognizing
            }

        now = time.time()
        
        if (now - self._last_run >= self.interval) and not self._is_recognizing:
            self._last_run = now
            self._is_recognizing = True
            
            # Pass a copy of the entire frame instead of cropping.
            # DeepFace's internal retinaface detector needs the whole context 
            # to properly detect and align the face.
            frame_roi = frame.copy()
            
            if frame_roi.size > 0:
                threading.Thread(target=self._recognize_worker, args=(frame_roi,), daemon=True).start()

        return {
            "identity_name": self._identity_name,
            "identity_confidence_distance": round(self._identity_distance, 3),
            "identity_is_thinking": self._is_recognizing
        }
    
recognize_face = FaceRecognizer()