import cv2
import numpy as np

class CameraCapture:
    """SOURCE -> yields raw frames from the camera."""
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def get_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def close(self):
        if self.cap.isOpened():
            self.cap.release()