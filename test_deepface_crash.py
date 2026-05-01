import cv2
import time
from face_recognizer import FaceRecognizer

recognizer = FaceRecognizer()
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
if ok:
    print('Testing deepface via the recognizer worker directly...')
    recognizer._recognize_worker(frame)
    print(f'Done! Result: {recognizer._identity_name}')
cap.release()
