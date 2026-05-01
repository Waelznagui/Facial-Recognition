import numpy as np
from inference import get_model
from dotenv import load_dotenv
import cv2
import os

load_dotenv()
api_key = os.getenv("api_key")
class DistractionDetector:
    """DETECTOR -> is the user distracted?"""
    def __init__(self):
        self.class_filter = ["cell phone", "cell_phone", "mobile"]
        self._distraction_frames = 15
        self.DISTRACTION_THRESHOLD = 30 # roughly 1 second
        self._model = get_model(model_id="mobile-detection-l2iov-nxlqw/1", api_key=api_key) # Lazy load model when needed
    
    def process(self, frame: np.ndarray, context: dict) -> dict:
        results = self._model.infer(frame, confidence=0.15)[0]
        
        predicted_classes = [p.class_name for p in results.predictions]
        has_distraction_item = any(item in self.class_filter for item in predicted_classes)
        
        if has_distraction_item:
            # We add +5 so it triggers fast when it sees a phone
            self._distraction_frames = min(self._distraction_frames + 5, self.DISTRACTION_THRESHOLD)
        else:
            # We subtract -2 so it drains the frames normally when no phone is seen
            self._distraction_frames = max(0, self._distraction_frames - 2)
            
        distracted = self._distraction_frames >= self.DISTRACTION_THRESHOLD
            
        return {
            "distraction_is_distracted": distracted,
        }
#distraction_detector = DistractionDetector()
if __name__ == "__main__":
    cap=cv2.VideoCapture(0)
    detector = DistractionDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.process(frame, {})
        print(result)