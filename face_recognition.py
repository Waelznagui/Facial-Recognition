from deepface import DeepFace 
import os 
import cv2
import time


MODEL="ArcFace"
DETECTOR="retinaface"
Threshold=0.6

def recognize(frame)->str:
    dict={}
    results =DeepFace.find(frame, db_path = "faces_db", model_name=MODEL, detector_backend=DETECTOR, enforce_detection=False)
    if not results or results[0].empty:
        dict={
        "name": "Unknown",
        "x": 0,
        "y": 0,
        "w": 0,
        "h": 0
    }
        return dict
    best_match =results[0].iloc[0]
    
    identity_path = best_match['identity']
    name = os.path.splitext(os.path.basename(identity_path))[0]
    x_db = best_match['source_x']
    y_db = best_match['source_y']
    w_db = best_match['source_w']
    h_db = best_match['source_h']
    distance= best_match["distance"]
   
    if distance > Threshold:
        name ="Unknown"
    dict={
        "name": name,
        "x": x_db,
        "y": y_db,
        "w": w_db,
        "h": h_db
    }
    return dict
def main():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    label = "Starting..."
    last_run = 0.0
    run_every_seconds = 0.5
    x_db, y_db, w_db, h_db = 0, 0, 0, 0 
    while True:
        ok, frame=cap.read()
        if not ok:
            print("Error: Could not read frame.")
            break

        now = time.time()
        if now - last_run >= run_every_seconds:
            last_run = now
            dict = recognize(frame)
            label = dict["name"]
            x_db = dict['x']
            y_db = dict['y']
            w_db = dict['w']
            h_db = dict['h']
        if label != "Unknown":
                # Draw rectangle around the recognized face
            cv2.rectangle(frame, (x_db, y_db), (x_db + w_db, y_db + h_db), (255, 0, 0), 2)
        
        # Display the label on the frame
        cv2.putText(frame, f"Person: {label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the video feed
        cv2.imshow("Face Recognition", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("temp.jpg"):
        os.remove("temp.jpg")

if __name__ == "__main__":
    main()






