import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    """DETECTOR -> is a face present? where?"""
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        self.detector = vision.FaceDetector.create_from_options(options)

    def process(self, frame: np.ndarray, context: dict) -> dict:
        """
        Uses MediaPipe Face Detection to quickly find the face bounding box.
        Runs every frame without blocking.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        
        if not results.detections:
            return {"face_present": False}
        
        # Get the most prominent face
        bbox = results.detections[0].bounding_box
        
        return {
            "face_present": True,
            "face_x": max(0, bbox.origin_x),
            "face_y": max(0, bbox.origin_y),
            "face_w": max(0, bbox.width),
            "face_h": max(0, bbox.height)
        }