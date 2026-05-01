import time
import json

class EventEmitter:
    """PUBLISHER -> fires typed events to subscribers"""
    def __init__(self, throttle_seconds=2.0):
        self.last_emitted_state = None
        self.last_emit_time = 0
        self.throttle_seconds = throttle_seconds

    def publish(self, state: dict):
        """
        Emits the state to Arduino/Hardware/Chatbot.
        (For now, mocked as terminal JSON prints on state changes).
        """
        current_user_state = state.get("user_state_label")
        
        # Only log to terminal if the core state changed, or every N seconds to prevent spam
        now = time.time()
        if current_user_state != self.last_emitted_state or (now - self.last_emit_time) > self.throttle_seconds:
            self.last_emitted_state = current_user_state
            self.last_emit_time = now
            
            # Print a clean, formatted payload imitating a system event
            emit_payload = {
                "user_state": current_user_state,
                "identity": state.get("identity_name", "Unknown"),
                "fatigue": state.get("fatigue_level", "Unknown"),
                "distracted": state.get("distraction_is_distracted", False)
            }
            print(f"[EVENT EMITTER] -> {json.dumps(emit_payload)}")