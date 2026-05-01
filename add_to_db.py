import cv2
import os 
from deepface import DeepFace
import threading

# Match the detector used in the main application
DETECTOR = "retinaface"

# Global flags for thread communication
save_requested = False
quit_requested = False

def listen_for_input():
    global save_requested, quit_requested
    while not quit_requested:
        user_input = input().strip().lower()
        if user_input == 's':
            save_requested = True
        elif user_input == 'q':
            quit_requested = True

def add_from_camera():
    global save_requested, quit_requested
    name = input("Enter name for the new face (a folder will be created): ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    save_dir = os.path.join("faces_db", name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return  
        
    count = 0
    print("\nInstructions:")
    print(f" - Enrolling: {name}")
    print(" - Look at the camera window.")
    print(" - Click back HERE IN THE TERMINAL and press 's' + ENTER to save a photo (save 3-5 images for better accuracy).")
    print(" - Click back HERE IN THE TERMINAL and press 'q' + ENTER to quit when done.\n")

    # Start terminal input listener in a separate thread
    input_thread = threading.Thread(target=listen_for_input, daemon=True)
    input_thread.start()

    # Load a fast Haar Cascade just for the visual guide (not the actual saving engine)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: Could not read frame.")
            break
            
        display_frame = frame.copy()
        
        # Fast detection for visual sign
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(detected_faces) > 0:
            cv2.putText(display_frame, "READY TO SAVE! (Face Detected)", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for (x, y, w, h) in detected_faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "FINDING FACE...", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
        cv2.putText(display_frame, f"Person: {name} | Saved: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
        cv2.imshow("Add to Database", display_frame)
        cv2.waitKey(30) # Needed just to update the window, we don't read keys from it anymore
        
        if save_requested:
            save_requested = False
            # Verify a face is actually present *before* saving
            try:
                print("Processing... Please wait a second.")
                # enforce_detection=True throws an exception if no face is found
                faces = DeepFace.extract_faces(img_path=frame, detector_backend=DETECTOR, enforce_detection=True)
                if faces:
                    img_path = os.path.join(save_dir, f"{name}_{count}.jpg")
                    cv2.imwrite(img_path, frame)
                    print(f"✅ Saved {img_path} to database. Get ready for the next one!")
                    count += 1
            except ValueError:
                print("❌ No face detected! Please ensure your face is clearly visible.")
                
        if quit_requested:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Saved {count} images for {name}.")

def main():
    add_from_camera()

if __name__ == "__main__":
    main()
