from inference import get_model
from dotenv import load_dotenv
import cv2
import os

load_dotenv()
api_key = os.getenv("api_key")

# Load model once (downloads weights locally)
model = get_model(model_id="mobile-detection-l2iov-nxlqw/1", api_key=api_key)

# Check your actual class names on Roboflow Universe — adjust if needed
class_filter = ["person", "cell phone", "cell_phone", "mobile"]

cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Error: Could not read frame.")
        break

    # Run inference directly on the numpy frame — no temp file needed
    results = model.infer(frame, confidence=0.15)[0]

    for pred in results.predictions:
        class_name = pred.class_name

        if class_name not in class_filter:
            continue

        # pred.x/y are center coords, pred.width/height are dimensions
        x1 = int(pred.x - pred.width / 2)
        y1 = int(pred.y - pred.height / 2)
        x2 = int(pred.x + pred.width / 2)
        y2 = int(pred.y + pred.height / 2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{class_name}: {pred.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Mobile Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()