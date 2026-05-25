import cv2
import time

class HUDManager:
    def __init__(self, regions):
        self.panel_open = False
        self.panel_anim = 0.0
        self.panel_width = 350 
        self.tabs = ["Occlusion", "Kinematics", "Settings"]
        self.active_tab = "Kinematics"
        self.regions = regions
        
        self.max_trackers = 1            
        self.view_tracker_idx = 0        
        self.assignments = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4} 
        
        self.click_zones = [] 
        self.next_click_zones = []
        
        #* TOGGLES
        self.show_face_labels = True
        self.show_detected_emotion = True
        self.model_loaded = False 

        #* COUNTER TOGGLES
        self.show_face_counter = True
        self.show_tracker_counter = True
        self.show_fps_counter = True
    
        self.last_detected_count = 0 
        
        self.region_colors = {
            "Right Eyebrow": (0, 200, 255), "Left Eyebrow": (0, 200, 255),
            "Right Eye": (0, 255, 150), "Left Eye": (0, 255, 150),
            "Right Iris": (50, 255, 255), "Left Iris": (50, 255, 255),
            "Nose Bridge": (200, 200, 0), "Nose Tip": (200, 220, 0),
            "Nostrils": (180, 200, 0), "Upper Lip": (0, 80, 255),
            "Lower Lip": (0, 100, 255), "Jaw Line": (180, 180, 180),
            "Right Cheek": (140, 100, 220), "Left Cheek": (140, 100, 220),
            "Forehead": (100, 160, 240)
        }

        self.esc_state = 'idle'
        self.esc_down_t = 0
        self.esc_last_seen_t = 0
        self.esc_fill_t = 0
        self.ESC_WAIT_S = 0.65
        self.ESC_FILL_S = 1.35

    def handle_click(self, x, y):
        for (x1, y1, x2, y2, action, param) in self.click_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "open_panel":
                    self.panel_open = True
                elif action == "set_tab":
                    self.active_tab = param
                elif action == "set_subtab":
                    self.view_tracker_idx = param
                elif action == "close_panel":
                    self.panel_open = False
                elif action == "toggle_show_labels":
                    self.show_face_labels = not self.show_face_labels
                elif action == "toggle_show_emotions":
                    if self.model_loaded:
                        self.show_detected_emotion = not self.show_detected_emotion
                elif action == "toggle_face_counter":
                    self.show_face_counter = not self.show_face_counter
                elif action == "toggle_tracker_counter":
                    self.show_tracker_counter = not self.show_tracker_counter
                elif action == "toggle_fps_counter":
                    self.show_fps_counter = not self.show_fps_counter
                elif action == "adj_max":
                    new_max = max(1, min(5, self.max_trackers + param))
                    if new_max > self.max_trackers:
                        used = [self.assignments[i] for i in range(self.max_trackers)]
                        for possible_face in range(max(1, self.last_detected_count)):
                            if possible_face not in used:
                                self.assignments[self.max_trackers] = possible_face
                                break
                    self.max_trackers = new_max
                    if self.view_tracker_idx >= self.max_trackers:
                        self.view_tracker_idx = self.max_trackers - 1
                elif action == "adj_assign":
                    t_idx, direction = param
                    if self.last_detected_count <= 0:
                        self.assignments[t_idx] = 0
                    else:
                        active_other = [self.assignments[i] for i in range(self.max_trackers) if i != t_idx]
                        current = self.assignments[t_idx]
                        
                        if current >= self.last_detected_count:
                            current = -1 if direction == 1 else self.last_detected_count
                            
                        for _ in range(self.last_detected_count):
                            current = (current + direction) % self.last_detected_count
                            if current not in active_other:
                                self.assignments[t_idx] = current
                                break
                return 

    def handle_input(self, key):
        now = time.time()
        should_quit = False
        
        if key == 8 or key == 113: 
            should_quit = True
        
        if key == 27:
            self.esc_last_seen_t = now
            if self.esc_state == 'idle':
                self.esc_state = 'waiting'
                self.esc_down_t = now
        
        is_down = (now - self.esc_last_seen_t < 0.2)
        
        if is_down:
            if self.esc_state == 'waiting' and (now - self.esc_down_t) >= self.ESC_WAIT_S:
                self.esc_state = 'filling'
                self.esc_fill_t = now
            elif self.esc_state == 'filling':
                fill_ratio = min((now - self.esc_fill_t) / self.ESC_FILL_S, 1.0)
                if fill_ratio >= 1.0:
                    should_quit = True
        else:
            if self.esc_state == 'waiting':
                self.panel_open = not self.panel_open
                self.esc_state = 'idle'
            elif self.esc_state == 'filling':
                self.esc_state = 'idle'
        
        if self.panel_open:
            if key == ord('1'): self.active_tab = "Occlusion"
            if key == ord('2'): self.active_tab = "Kinematics"
            if key == ord('3'): self.active_tab = "Settings"
        
        return should_quit

    def _add_zone(self, x, y, w, h, action, param=None):
        self.next_click_zones.append((x, y, x+w, y+h, action, param))

    def _draw_bento_icon(self, frame, h, w):
        cx, cy = w - 45, h - 45
        
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), 28, (15, 15, 20), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        active = (self.esc_state != 'idle')
        fill_ratio = 0.0
        
        if self.esc_state == 'filling':
            fill_ratio = min((time.time() - self.esc_fill_t) / self.ESC_FILL_S, 1.0)

        base_col = (120, 120, 140) if not active else (200, 200, 220)
        fill_col = (80, 80, 255) 

        size, gap = 5, 8
        continuous_progress = fill_ratio * 9.0

        for i in range(-1, 2):
            for j in range(-1, 2):
                sx = cx + (i * gap) - (size // 2)
                sy = cy + (j * gap) - (size // 2)
                fill_order = (1 - j) * 3 + (i + 1)
                
                cv2.rectangle(frame, (sx, sy), (sx + size, sy + size), base_col, -1)
                
                box_fill = max(0.0, min(1.0, continuous_progress - fill_order))
                if box_fill > 0:
                    fill_h = max(1, int(size * box_fill)) 
                    cv2.rectangle(frame, (sx, sy + size - fill_h), (sx + size, sy + size), fill_col, -1)

        cv2.putText(frame, "ESC", (cx - 12, cy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        if not self.panel_open:
            self._add_zone(cx - 40, cy - 40, 80, 80, "open_panel")
        else:
            self._add_zone(cx - 40, cy - 40, 80, 80, "close_panel")

    def _draw_toggle(self, frame, x, y, label, is_on, action_name, disabled=False):
        """Helper to draw clean UI switches with disabled states"""
        txt_col = (100, 100, 100) if disabled else (200, 200, 200)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_col, 1)
        radius = 9
        pill_x = x + 230
        pill_y = y - 4
        pill_w = 24 
        
        if disabled:
            bg_col = (40, 40, 50)
            knob_col = (80, 80, 90)
        else:
            bg_col = (0, 200, 50) if is_on else (80, 80, 100)
            knob_col = (255, 255, 255)
        
        cv2.circle(frame, (pill_x, pill_y), radius, bg_col, -1, cv2.LINE_AA)
        cv2.circle(frame, (pill_x + pill_w, pill_y), radius, bg_col, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (pill_x, pill_y - radius), (pill_x + pill_w, pill_y + radius), bg_col, -1)
        
        knob_x = pill_x + pill_w if is_on else pill_x
        cv2.circle(frame, (knob_x, pill_y), radius - 2, knob_col, -1, cv2.LINE_AA)
        
        if not disabled:
            self._add_zone(pill_x - 20, pill_y - 20, pill_w + 40, 40, action_name)

    def draw(self, frame, h, w, fps, occ_data_list, kin_data_list, detected_count):
        self.last_detected_count = detected_count 
        self.next_click_zones = [] 
        
        status_parts = []
        if self.show_face_counter:
            status_parts.append(f"Detected Faces: {detected_count}")
        if self.show_tracker_counter:
            status_parts.append(f"Trackers Active: {self.max_trackers}")
        if self.show_fps_counter:
            status_parts.append(f"FPS: {int(fps)}")
            
        if status_parts:
            status_text = " | ".join(status_parts)
            cv2.putText(frame, status_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 120), 2, cv2.LINE_AA)

        target = 1.0 if self.panel_open else 0.0
        self.panel_anim += (target - self.panel_anim) * 0.2
        
        if self.panel_anim < 0.01:
            self._draw_bento_icon(frame, h, w)
            self.click_zones = self.next_click_zones 
            return

        pw = int(self.panel_width * self.panel_anim)
        x0 = w - pw
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, 0), (w, h), (18, 18, 30), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        y_tab = 30
        for i, tab in enumerate(self.tabs):
            col = (200, 200, 255) if self.active_tab == tab else (80, 80, 100)
            txt = f"[{i+1}] {tab}"
            cv2.putText(frame, txt, (x0 + 20 + (i*105), y_tab), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            self._add_zone(x0 + 10 + (i*105), y_tab - 20, 95, 30, "set_tab", tab)

        y = 60
        if self.active_tab in ["Occlusion", "Kinematics"] and self.max_trackers > 1:
            for i in range(self.max_trackers):
                col = (0, 255, 150) if self.view_tracker_idx == i else (80, 80, 100)
                cx_tab = x0 + 20 + (i*50)
                cv2.putText(frame, f"Trk {i+1}", (cx_tab, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
                self._add_zone(cx_tab - 5, y - 20, 50, 30, "set_subtab", i)
            y += 30

        if self.active_tab == "Settings":
            cv2.putText(frame, "Max Tracked Faces:", (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, "<", (x0 + 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,200), 2)
            cv2.putText(frame, str(self.max_trackers), (x0 + 230, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, ">", (x0 + 260, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,200), 2)
            self._add_zone(x0 + 180, y - 20, 40, 35, "adj_max", -1)
            self._add_zone(x0 + 250, y - 20, 40, 35, "adj_max", 1)
            
            y += 40 

            cv2.putText(frame, "- Face Assignments -", (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            y += 30
            for i in range(self.max_trackers):
                face_id = self.assignments[i]
                cv2.putText(frame, f"Tracker {i+1} maps to:", (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
                cv2.putText(frame, "<", (x0 + 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                cv2.putText(frame, f"Face {face_id}", (x0 + 220, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                cv2.putText(frame, ">", (x0 + 280, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                self._add_zone(x0 + 180, y - 20, 40, 35, "adj_assign", (i, -1))
                self._add_zone(x0 + 270, y - 20, 40, 35, "adj_assign", (i, 1))
                y += 35
                
            y += 10
            
            #* OVERLAYS SECTION
            cv2.putText(frame, "- Overlays -", (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            y += 30
            self._draw_toggle(frame, x0 + 20, y, "Display Face Labels:", self.show_face_labels, "toggle_show_labels")
            y += 30 
            
            #! Force the toggle off and gray it out if the model is missing
            if not self.model_loaded:
                self.show_detected_emotion = False
                
            self._draw_toggle(frame, x0 + 20, y, "Display Detected Emotion:", self.show_detected_emotion, "toggle_show_emotions", disabled=not self.model_loaded)
            y += 40
            
            #* COUNTERS SECTION
            cv2.putText(frame, "- Display Counters -", (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
            y += 30
            self._draw_toggle(frame, x0 + 20, y, "Face Counter:", self.show_face_counter, "toggle_face_counter")
            y += 30
            self._draw_toggle(frame, x0 + 20, y, "Tracker Counter:", self.show_tracker_counter, "toggle_tracker_counter")
            y += 30
            self._draw_toggle(frame, x0 + 20, y, "FPS Counter:", self.show_fps_counter, "toggle_fps_counter")
            
        else:
            occ_data = occ_data_list[self.view_tracker_idx]
            kin_data = kin_data_list[self.view_tracker_idx]
            has_data = len(kin_data) > 0
            
            for name in self.regions:
                if not has_data:
                    cv2.putText(frame, f"{name}: No Face Detected", (x0 + 20, y-2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 255), 1, cv2.LINE_AA)
                else:
                    occ = occ_data.get(name, False)
                    if self.active_tab == "Occlusion":
                        col = (80, 80, 255) if occ else (100, 200, 100)
                        cv2.circle(frame, (x0 + 20, y-4), 4, col, -1, cv2.LINE_AA)
                        cv2.putText(frame, f"{name}: {'OCCLUDED' if occ else 'visible'}", (x0 + 35, y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
                    else: 
                        if occ:
                            cv2.putText(frame, f"{name}: Paused (Occluded)", (x0 + 20, y-2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 255), 1, cv2.LINE_AA)
                        else:
                            val = kin_data.get(name, 0.0)
                            abs_val = min(abs(val), 1.0)
                            bar_w = int((pw - 100) * abs_val)
                            
                            bar_color = (0, 255, 0) if val >= 0 else (0, 255, 255)
                            
                            cv2.rectangle(frame, (x0 + 20, y+2), (x0 + 20 + bar_w, y+5), bar_color, -1)
                            cv2.putText(frame, f"{name}: {val:.3f}", (x0 + 20, y-2), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
                y += 25
                if y > h - 50: break

        self._draw_bento_icon(frame, h, w)
        
        cv2.putText(frame, "(hold ESC or press Backspace to quit)", (x0 + 20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 140), 1, cv2.LINE_AA)

        self.click_zones = self.next_click_zones