import tkinter as tk
from tkinter import messagebox
import os
import shutil
import subprocess
import sys

#* ─────────────────────────────────────────────────────────────────
#* UNIX INSTALLATION MANAGER
#* ─────────────────────────────────────────────────────────────────
INSTALL_DIR = os.path.expanduser("~/.emoprosopon")
SYMLINK_PATH = "/usr/local/bin/eop"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class UnixManager(tk.Tk):
    def __init__(self, mode="install"):
        super().__init__()
        self.mode = mode
        self.title("EmoProsopon Manager")
        self.geometry("450x300")
        self.configure(bg="#0a0a0c")
        self.resizable(False, False)
        
        self.build_ui()

    def build_ui(self):
        title_text = "EmoProsopon Setup" if self.mode == "install" else "EmoProsopon Uninstaller"
        
        tk.Label(self, text=title_text, font=("Courier", 18, "bold"), fg="#00ffcc", bg="#0a0a0c").pack(pady=20)
        
        desc = ("Install EmoProsopon to your system.\nThis will create a global 'eop' command." 
                if self.mode == "install" 
                else "Remove EmoProsopon and all related\nenvironment variables from your system.")
                
        tk.Label(self, text=desc, font=("Courier", 10), fg="#a9a9a9", bg="#0a0a0c", justify="center").pack(pady=10)

        self.status_label = tk.Label(self, text="", font=("Courier", 9), fg="#ffffff", bg="#0a0a0c")
        self.status_label.pack(pady=10)

        btn_text = "Install" if self.mode == "install" else "Uninstall"
        btn_cmd = self.install if self.mode == "install" else self.uninstall
        btn_color = "#00ffcc" if self.mode == "install" else "#ff3333"

        self.action_btn = tk.Button(self, text=btn_text, font=("Courier", 12, "bold"), 
                                    bg="#1a1a1f", fg=btn_color, activebackground="#2a2a30", 
                                    activeforeground=btn_color, relief="flat", command=btn_cmd)
        self.action_btn.pack(pady=10, ipadx=20, ipady=5)

    def log(self, msg, color="#ffffff"):
        self.status_label.config(text=msg, fg=color)
        self.update()

    def install(self):
        self.action_btn.config(state="disabled")
        try:
            self.log("Copying files to ~/.emoprosopon...")
            if os.path.exists(INSTALL_DIR):
                shutil.rmtree(INSTALL_DIR)
            
            # Copy everything except the installer GUI itself
            shutil.copytree(REPO_ROOT, INSTALL_DIR, ignore=shutil.ignore_patterns('installers', '.vscode', '__pycache__'))
            
            os.chmod(os.path.join(INSTALL_DIR, "eop.py"), 0o755)
            
            self.log("Requesting sudo for global symlink...", "#00ffcc")
            # Ask for sudo to create the symlink
            symlink_cmd = f"sudo ln -sf {os.path.join(INSTALL_DIR, 'eop.py')} {SYMLINK_PATH}"
            subprocess.run(["xterm", "-e", symlink_cmd], check=False) # Attempts to open terminal for password
            
            messagebox.showinfo("Success", "EmoProsopon Installed!\nRun 'eop --setup' in your terminal.")
            self.destroy()
            
        except Exception as e:
            self.log(f"Error: {str(e)}", "#ff3333")
            self.action_btn.config(state="normal")

    def uninstall(self):
        self.action_btn.config(state="disabled")
        try:
            self.log("Removing installation directory...")
            if os.path.exists(INSTALL_DIR):
                shutil.rmtree(INSTALL_DIR)
                
            self.log("Removing global command...")
            subprocess.run(["sudo", "rm", "-f", SYMLINK_PATH], check=False)
            
            messagebox.showinfo("Success", "EmoProsopon has been completely removed.")
            self.destroy()
            
        except Exception as e:
            self.log(f"Error: {str(e)}", "#ff3333")
            self.action_btn.config(state="normal")

if __name__ == "__main__":
    run_mode = "install"
    if len(sys.argv) > 1 and sys.argv[1] == "--uninstall":
        run_mode = "uninstall"
        
    app = UnixManager(mode=run_mode)
    app.mainloop()