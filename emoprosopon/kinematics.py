import numpy as np
from collections import deque

class KinematicManager:
    def __init__(self, regions, buffer_size=10):
        self.activity = {r: 0.0 for r in regions}
        self.history = {r: deque(maxlen=buffer_size) for r in regions}
        
        self.prev_yaw = 0.5 
        self.current_yaw_penalty = 0.0
        
        #! These regions physically move "Down", so we invert them to show contrast
        self.invert_regions = ["Lower Lip", "Jaw Line", "Nose Tip", "Left Cheek", "Right Cheek"]

    def compute_local_coordinates(self, face_landmarks, w, h):
        def to_px(lm):
            return np.array([lm.x * w, lm.y * h, lm.z * w])

        n_x = face_landmarks[1].x   
        l_x = face_landmarks[234].x 
        r_x = face_landmarks[454].x 

        dist_left = abs(n_x - l_x)
        dist_right = abs(r_x - n_x)
        current_yaw = dist_left / (dist_left + dist_right + 1e-6) 
        
        yaw_velocity = abs(current_yaw - self.prev_yaw)
        self.prev_yaw = current_yaw
        self.current_yaw_penalty = yaw_velocity * 5.0

        anchor_nose = to_px(face_landmarks[168])      #? Top of nose (Bridge)
        anchor_nose_tip = to_px(face_landmarks[1])    #? Tip of nose
        anchor_l_eye = to_px(face_landmarks[33])      #? Left Eye
        anchor_r_eye = to_px(face_landmarks[263])     #? Right Eye

        scale = np.linalg.norm(anchor_nose - anchor_nose_tip)
        if scale == 0: scale = 1.0

        local_points = {}
        for i, lm in enumerate(face_landmarks):
            p = to_px(lm)
            
            d1 = np.linalg.norm(p - anchor_nose) / scale
            d2 = np.linalg.norm(p - anchor_l_eye) / scale
            d3 = np.linalg.norm(p - anchor_r_eye) / scale
            
            local_points[i] = np.array([d1, d2, d3])

        return local_points

    def update(self, rname, local_points_list):
        if not local_points_list:
            self.activity[rname] = 0.0
            return

        avg_signature = np.mean(local_points_list, axis=0)
        self.history[rname].append(avg_signature)

        if len(self.history[rname]) == self.history[rname].maxlen:
            start_sig = self.history[rname][0]
            current_sig = self.history[rname][-1]
            
            #* 1. Calculate the True Magnitude (High Sensitivity restored)
            movement_magnitude = np.linalg.norm(current_sig - start_sig)
            
            #* 2. Calculate the Direction (+ or -)
            #? If current distances are LARGER, the mean is negative (Expanding)
            direction = np.sign(np.mean(start_sig - current_sig))
            if direction == 0: direction = 1.0
            
            #* 3. Combine them
            signed_movement = movement_magnitude * direction
            
            #* 4. Invert the lower face
            if rname in self.invert_regions:
                signed_movement *= -1.0
            
            if rname in ["Jaw Line", "Upper Lip", "Lower Lip"]:
                base_gate = 0.018 
            elif rname in ["Right Cheek", "Left Cheek"]:
                base_gate = 0.012
            else:
                base_gate = 0.008 
            
            active_gate = base_gate + self.current_yaw_penalty
            
            if abs(signed_movement) < active_gate:
                activity = 0.0
            else:
                activity = signed_movement - (np.sign(signed_movement) * active_gate)
            
            clamped_val = np.clip(activity * 15.0, -1.0, 1.0)
            self.activity[rname] = clamped_val