import tkinter as tk
from tkinter import filedialog
import mss
import math
import random
import landmarks

#* ─────────────────────────────────────────────────────────────────
#* FALLING LINES ANIMATION
#* ─────────────────────────────────────────────────────────────────
class FallingLinesCanvas(tk.Canvas):
    DENSITY = 1 / 5000

    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#000000", highlightthickness=0, **kwargs)
        self.lines = []
        self.bind("<Configure>", self._on_resize)
        self.after(100, self._init_and_start)

    def _on_resize(self, event):
        self._init_lines(event.width, event.height)

    def _init_and_start(self):
        w = self.winfo_width()
        h = self.winfo_height()
        self._init_lines(w, h)
        self._animate()

    def _init_lines(self, w, h):
        count = max(1, int(w * h * self.DENSITY))
        self.lines = [self._create_line(w, h) for _ in range(count)]

    def _create_line(self, w=None, h=None):
        w = w or self.winfo_width()
        h = h or self.winfo_height()
        return {
            "x":            random.uniform(0, w),
            "y":            random.uniform(-100, h),
            "length":       random.uniform(20, 100),
            "width":        random.uniform(0.5, 2.5),
            "opacity":      0.0,
            "targetOpacity":random.uniform(0.5, 1.0),
            "speed":        random.uniform(0.3, 0.8),
            "fadeSpeed":    random.uniform(0.002, 0.007),
            "fadingIn":     True,
        }

    def _animate(self):
        self.delete("line")
        w = self.winfo_width()
        h = self.winfo_height()

        for i, ln in enumerate(self.lines):
            #* Draw
            alpha = int(ln["opacity"] * 255)
            color = f"#{alpha:02x}{alpha:02x}{alpha:02x}"
            self.create_line(
                ln["x"], ln["y"],
                ln["x"], ln["y"] + ln["length"],
                fill=color, width=ln["width"], tags="line"
            )
            # Update position
            ln["y"] += ln["speed"]

            #* Fade in / out
            if ln["fadingIn"]:
                ln["opacity"] = min(ln["opacity"] + ln["fadeSpeed"], ln["targetOpacity"])
                if ln["opacity"] >= ln["targetOpacity"]:
                    ln["fadingIn"] = False
            else:
                ln["opacity"] = max(ln["opacity"] - ln["fadeSpeed"], 0.0)

            # Reset when done
            if ln["opacity"] <= 0 or ln["y"] > h:
                self.lines[i] = self._create_line(w, h)
                self.lines[i]["y"] = random.uniform(-100, 0)  # restart from top

        self.after(16, self._animate)  # ~60fps


#* ─────────────────────────────────────────────────────────────────
#* STYLED HOVER BUTTON
#* ─────────────────────────────────────────────────────────────────
class StyledButton(tk.Canvas):
    DEFAULT_BORDER = "#a9a9a9"
    HOVER_BORDER   = "#ffffff"
    DEFAULT_TEXT   = "#a9a9a9"
    HOVER_TEXT     = "#ffffff"
    HOVER_BG       = "#36454f"
    HOVER_BG_NEG   = "#7c0a02"
    RADIUS         = 6

    def __init__(self, master, text, command=None, width=160, height=44, font_size=13, **kwargs):
        kwargs.pop("width", None)
        kwargs.pop("height", None)
        super().__init__(master, width=width, height=height, bg="#000000", highlightthickness=0, **kwargs)

        self._text    = text
        self._command = command
        self.width    = width
        self.height   = height
        self._fs      = font_size
        self._hovered = False

        self.after(0, self._draw)
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        outline = kwargs.get("outline", "#ffffff")
        width   = kwargs.get("width", 1)
        fill    = kwargs.get("fill", "")

        self.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, style="arc", outline=outline, width=width)
        self.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, style="arc", outline=outline, width=width)
        self.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, style="arc", outline=outline, width=width)
        self.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, style="arc", outline=outline, width=width)
        self.create_line(x1+r, y1, x2-r, y1, fill=outline, width=width)
        self.create_line(x1+r, y2, x2-r, y2, fill=outline, width=width)
        self.create_line(x1, y1+r, x1, y2-r, fill=outline, width=width)
        self.create_line(x2, y1+r, x2, y2-r, fill=outline, width=width)

        if fill:
            self.create_rectangle(x1+r, y1, x2-r, y2, fill=fill, outline="")
            self.create_rectangle(x1, y1+r, x2, y2-r, fill=fill, outline="")

    def _draw(self):
        self.delete("all")
        w, h, r = self.width, self.height, self.RADIUS
        bg     = self.HOVER_BG if self._hovered and self._text != "Exit" else self.HOVER_BG_NEG if self._hovered and self._text == "Exit" else "#000000"
        border = self.HOVER_BORDER if self._hovered else self.DEFAULT_BORDER
        color  = self.HOVER_TEXT   if self._hovered else self.DEFAULT_TEXT

        #* Background fill
        self.create_rectangle(r, 0, w-r, h, fill=bg, outline="")
        self.create_rectangle(0, r, w, h-r, fill=bg, outline="")
        self.create_oval(0, 0, 2*r, 2*r, fill=bg, outline="")
        self.create_oval(w-2*r, 0, w, 2*r, fill=bg, outline="")
        self.create_oval(0, h-2*r, 2*r, h, fill=bg, outline="")
        self.create_oval(w-2*r, h-2*r, w, h, fill=bg, outline="")

        #* Border
        self._rounded_rect(1, 1, w-1, h-1, r, outline=border, width=2)

        #* Label
        self.create_text(w//2, h//2, text=self._text, fill=color, font=("Helvetica", self._fs))

    def _on_enter(self, _):
        self._hovered = True
        self._draw()

    def _on_leave(self, _):
        self._hovered = False
        self._draw()

    def _on_click(self, _):
        if self._command:
            self._command()


#* ─────────────────────────────────────────────────────────────────
#* SCALE HELPER 
#* ─────────────────────────────────────────────────────────────────
def scale(base_w=640, base_h=480, current_w=640, current_h=480):
    return max(0.5, math.log10((current_w * current_h) / (base_w * base_h)) + 1)


#* ─────────────────────────────────────────────────────────────────
#* ACTIONS  
#* ─────────────────────────────────────────────────────────────────
def launch_camera():
    root.destroy()
    landmarks.run_tracker(source_type="camera", source_val=0)

def launch_video():
    filepath = filedialog.askopenfilename(
        title="Select Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

    if filepath:
        #! Hide menu instead of destroying it
        root.withdraw()
        landmarks.run_tracker(source_type="video", source_val=filepath)
        root.deiconify()

def launch_screen():
    selection = monitor_var.get()
    monitor_index = int(selection.split(" ")[1])
    root.destroy()
    landmarks.run_tracker(source_type="screen", source_val=monitor_index)


#* ─────────────────────────────────────────────────────────────────
#* MAIN WINDOW
#* ─────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("EmoProsopon")
root.geometry("640x480")
root.minsize(400, 320)
root.configure(bg="#000000")

#* Background canvas (fills entire window) ──────────────────────
bg_canvas = FallingLinesCanvas(root)
bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

#? Title ─────────────────────────────────────────────────────────
title_label = tk.Label(
    root,
    text="EmoProsopon",
    font=("Georgia", 28, "bold"),
    fg="#ffffff",
    bg="#000000",
)
title_label.place(relx=0.5, rely=0.0, anchor="n", y=40)

#? Description ───────────────────────────────────────────────────
desc_label = tk.Label(
    root,
    text="Real-time facial kinematics\nand emotion recognition engine.",
    font=("Helvetica", 11),
    fg="#a9a9a9",
    bg="#000000",
    justify="center",
)
desc_label.place(relx=0.25, rely=1.0, anchor="s", y=-50)

#? ── Buttons ───────────────────────────────────────────────────────
BTN_W, BTN_H, BTN_GAP = 160, 40, 12

buttons_frame = tk.Frame(root, bg="#000000")
buttons_frame.place(relx=0.75, rely=1.0, anchor="s", y=-40)

with mss.MSS() as sct:
    num_monitors = len(sct.monitors) - 1
    monitor_list = [f"Monitor {i}" for i in range(1, num_monitors + 1)]

monitor_var = tk.StringVar(root, value=monitor_list[0] if monitor_list else "Monitor 1")

button_defs = [
    ("Live Camera",     launch_camera),
    ("Load Video",      launch_video),
    ("Capture Screen",  launch_screen),
    ("Exit",            root.destroy)
]

ui_buttons = [] # Store references to scale them later

for text, cmd in button_defs:
    btn = StyledButton(buttons_frame, text=text, command=cmd, width=BTN_W, height=BTN_H)
    btn.pack(pady=BTN_GAP // 2)
    ui_buttons.append(btn)

#* ─────────────────────────────────────────────────────────────────
#* DYNAMIC RESIZE HANDLER
#* ─────────────────────────────────────────────────────────────────
def _on_root_resize(event):
    # Only react to the main window resizing, ignore child widget events
    if event.widget == root:
        w, h = event.width, event.height
        if w < 10 or h < 10: return # Prevent math errors on minimize
        
        s = scale(640, 480, w, h)
        
        # Scale Labels
        title_label.config(font=("Georgia", int(28 * s), "bold"))
        desc_label.config(font=("Helvetica", int(11 * s)))
        
        # Scale Buttons
        for btn in ui_buttons:
            btn.width = int(BTN_W * s)
            btn.height = int(BTN_H * s)
            btn._fs = int(13 * s)
            btn.config(width=btn.width, height=btn.height)
            btn._draw() # Force redraw with new dimensions

root.bind("<Configure>", _on_root_resize)
root.mainloop()