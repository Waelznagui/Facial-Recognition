from deepface import DeepFace 
import os 
import cv2
import time
import threading

# Match the detector used in the main application
MODEL = "ArcFace"
# Note: For a Raspberry Pi or low-end device, change 'retinaface' to 'opencv' or 'hog' for speed.
DETECTOR = "retinaface" 
# DeepFace calculates this automatically if you specify distance_metric, but for ArcFace + Cosine, 0.68 is optimal.
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.68

# --- MULTITHREADING SHARED VARIABLES ---
recognition_result = None
is_recognizing = False

def recognize_worker(frame_copy):
    """Background thread to prevent DeepFace from freezing the camera window."""
    global recognition_result, is_recognizing
    try:
        res = recognize(frame_copy)
        recognition_result = res
    except Exception as e:
        print(f"\n[Thread Error] Something crashed: {e}")
    finally:
        is_recognizing = False

def recognize(frame) -> dict:
    result_dict = {
        "name": "Unknown",
        "x": 0, "y": 0, "w": 0, "h": 0,
        "face_found": False
    }
    
    try:
        # silent=True prevents DeepFace from spamming the console every second
        results = DeepFace.find(
            frame, 
            db_path="faces_db", 
            model_name=MODEL, 
            detector_backend=DETECTOR, 
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            silent=True,
            anti_spoofing=False # Turned off for now to fix errors
        )
        
        if not results or results[0].empty:
            try:
                # We still want to see Unknown faces! Get their coordinates.
                faces = DeepFace.extract_faces(frame, detector_backend=DETECTOR, enforce_detection=False)
                if faces and len(faces) > 0 and 'facial_area' in faces[0]:
                    area = faces[0]['facial_area']
                    if area['w'] > 0 and area['h'] > 0:
                        result_dict['x'] = int(area['x'])
                        result_dict['y'] = int(area['y'])
                        result_dict['w'] = int(area['w'])
                        result_dict['h'] = int(area['h'])
                        result_dict['face_found'] = True
            except:
                pass
            return result_dict
            
        best_match = results[0].iloc[0]
        
        # DeepFace returns bounding box info. We convert them to integers.
        x_db = int(best_match['source_x'])
        y_db = int(best_match['source_y'])
        w_db = int(best_match['source_w'])
        h_db = int(best_match['source_h'])
        
        distance = best_match["distance"]
        
        # Determine name based on threshold
        name = "Unknown"
        if distance <= THRESHOLD:
            identity_path = best_match['identity']
            # Since we now use folders for multiple images (faces_db/Name/img.jpg):
            name = os.path.basename(os.path.dirname(identity_path))
            if name == "faces_db": # Fallback in case images are flat in the root folder
                name = os.path.splitext(os.path.basename(identity_path))[0]
        
        result_dict = {
            "name": name,
            "x": x_db,
            "y": y_db,
            "w": w_db,
            "h": h_db,
            "face_found": True
        }
    except ValueError:
        pass
    except Exception as e:
        print(f"\n[DeepFace Error] {e}")

    return result_dict

def get_tracker():
    """Attempt to initialize an OpenCV tracker."""
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        try:
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            print("Notice: 'opencv-contrib-python' not found. Bounding box tracking is disabled.")
            return None

def main():
    global recognition_result, is_recognizing
    
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    print("✅ Camera opened successfully. Loading DeepFace models... (This may take 10-20 seconds on the first run!)")
    
    label = "Waking up model..."
    # Set last_run to time.time() + 2.0 so the window has time to open and render BEFORE the heavy model freezes it
    last_run = time.time() + 2.0 
    # Increased to 1 second. The tracker will handle the rapid frames!
    run_every_seconds = 1.0 
    
    x_db, y_db, w_db, h_db = 0, 0, 0, 0 
    tracker = None
    tracking_active = False

    while True:
        ok, frame=cap.read()
        if not ok:
            print("Error: Could not read frame.")
            break

        now = time.time()
        
        # --- 1. HEAVY DEEP LEARNING INFERENCE (BACKGROUND THREAD) ---
        if now - last_run >= run_every_seconds and not is_recognizing:
            last_run = now
            is_recognizing = True
            
            # Start background thread so the GUI never freezes!
            threading.Thread(target=recognize_worker, args=(frame.copy(),), daemon=True).start()

        # Did the thread finish talking to DeepFace? Look at the result box!
        if recognition_result is not None:
            res = recognition_result
            recognition_result = None # empty the result box
            
            label = res["name"]
            if res["face_found"]:
                x_db, y_db, w_db, h_db = res['x'], res['y'], res['w'], res['h']
                
                # Initialize tracker on newly detected face
                tracker = get_tracker()
                if tracker is not None:
                    tracker.init(frame, (x_db, y_db, w_db, h_db))
                    tracking_active = True
                else:
                    tracking_active = True # Fallback to frozen bounding box
            else:
                tracking_active = False
                tracker = None

        # --- 2. LIGHTWEIGHT TRACKING (Foreground thread) ---
        elif tracking_active and tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                x_db, y_db, w_db, h_db = [int(v) for v in bbox]
            else:
                # Target lost (e.g. they turned their head too fast)
                tracking_active = False
                tracker = None

        # --- 3. DRAWING & DISPLAY ---
        if tracking_active:
            # Green for recognized, Red for unknown
            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x_db, y_db), (x_db + w_db, y_db + h_db), color, 2)
            
            # Draw label right above the box
            text_y = max(30, y_db - 10) # Ensures text doesn't go off-screen
            cv2.putText(frame, f"{label}", (x_db, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        
        if is_recognizing:
            # Tell the user that the AI is currently thinking in the background
            cv2.putText(frame, "Analyzing...", (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.imshow("Face Recognition", frame)
        
        # Increased to 30 to fix the freezing keyboard inputs we saw previously
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("temp.jpg"):
        try: os.remove("temp.jpg")
        except: pass

if __name__ == "__main__":
    main()






