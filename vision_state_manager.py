class VisionStateManager:
    """AGGREGATOR -> merges all detector outputs and applies state logic"""
    def __init__(self):
        self.shared_state = {}

    def merge(self, results: list) -> dict:
        """
        Flattens the list of detector output dicts into one master state.
        Determines the final 'user_state_label'.
        """
        self.shared_state.clear()
        
        for res in results:
            self.shared_state.update(res)
            
        # Determine overarching UI/System state
        # Hierarchy: Absent -> Fatigued -> Distracted -> Active
        final_state = "active"
        
        if not self.shared_state.get("face_present", False):
            final_state = "absent"
        elif self.shared_state.get("fatigue_level") in ["Mild", "Critical"]:
            final_state = "fatigued"
        elif self.shared_state.get("distraction_is_distracted", False):
            final_state = "distracted"
        
        self.shared_state["user_state_label"] = final_state
        return self.shared_state