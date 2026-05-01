import cv2
import time

# Import the architecture components
from camera_capture import CameraCapture
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from fatigue_detector import FatigueDetector
from distraction_detector import DistractionDetector
from vision_state_manager import VisionStateManager
from event_emitter import EventEmitter

def draw_hud(frame, state: dict):
    """(Optional Developer Vision) Draws the telemetry to the raw frame."""
    if not state.get("face_present"):
        cv2.putText(frame, "Waiting for face...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return
        
    x = state.get("face_x", 0)
    y = state.get("face_y", 0)
    w = state.get("face_w", 0)
    h = state.get("face_h", 0)
    name = state.get("identity_name", "")
    
    # Bounding box
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    # Identity
    cv2.putText(frame, f"ID: {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if state.get("identity_is_thinking"):
        cv2.putText(frame, "Comparing...", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Master State
    user_state = state.get("user_state_label", "unknown").upper()
    state_color = (0, 255, 0)
    if user_state == "FATIGUED": state_color = (0, 0, 255)
    if user_state == "DISTRACTED": state_color = (0, 165, 255)
    
    cv2.putText(frame, f"STATE: {user_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
    cv2.putText(frame, f"Fatigue: {state.get('fatigue_level')}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def main():
    print("Starting Vision Pipeline Orchestrator...")
    camera = CameraCapture(0)
    
    # Initialize all detectors according to contract
    detectors = [
        FaceDetector(),             # -> Gets geometry
        FaceRecognizer(),           # -> Gets identity (runs async)
        FatigueDetector(),          # -> Gets dropping eyelids/yawns 
        DistractionDetector()       # -> Checks if user is wandering
    ]
    
    state_manager = VisionStateManager()
    emitter = EventEmitter(throttle_seconds=3.0)

    try:
        for frame in camera.get_frames():
            context = {}
            
            # --- VISION LOOP ---
            # Run all detectors in precise order using the shared context
            for detector in detectors:
                result = detector.process(frame, context)
                if result:
                    context.update(result) # Instantly inject result so downstream detectors can use it
                    #print(context) # For debugging: See the evolving context in real-time as detectors add to it
            # --- AGGREGATION ---
            final_state = state_manager.merge([context])
            
            # --- PUBLISHING ---
            emitter.publish(final_state)
            
            # --- DISPLAY (Dev only) ---
            draw_hud(frame, final_state)
            cv2.imshow("Vision Core Orchestrator", frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
                
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()