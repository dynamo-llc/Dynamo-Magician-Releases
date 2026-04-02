import os
import sys
import threading
import time
from pathlib import Path
from PIL import Image
import torch

try:
    import customtkinter as ctk
    from tkinter import filedialog
except ImportError:
    print("Dependencies not found. Run: pip install customtkinter")
    sys.exit(1)

# Import the model pipeline
from generate_prequant import WanI2V_PreQuant, save_video

# --- Constants ---
APP_TITLE = "LingBot-World I2V GUI"
DEFAULT_SIZE = "480*832"
DEFAULT_STEPS = 40
DEFAULT_GUIDE = 5.0
DEFAULT_FRAMES = 81

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1000x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- State ---
        self.image_path = None
        self.action_path = None
        self.pipeline = None
        self.is_generating = False

        self._build_ui()
        
        self.log("Ready. Select an image to start.")

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main horizontal container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_columnconfigure(0, weight=3) # Left: Image
        self.main_container.grid_columnconfigure(1, weight=2) # Right: Controls

        # --- Left Side: Image Selection & Preview ---
        self.left_panel = ctk.CTkFrame(self.main_container, corner_radius=15, border_width=2, border_color="#3B8ED0")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(1, weight=1) # Preview takes the bottom space

        # 1. Action Button (Top)
        self.select_btn = ctk.CTkButton(self.left_panel, text="Step 1: Select Input Image", height=60, font=("Inter", 16, "bold"), command=self._select_input_image)
        self.select_btn.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

        # 2. Preview Area
        self.preview_container = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.preview_container.grid(row=1, column=0, sticky="nsew")
        self.preview_container.grid_columnconfigure(0, weight=1)
        self.preview_container.grid_rowconfigure(0, weight=1)

        self.drop_label = ctk.CTkLabel(self.preview_container, text="No Image Selected", font=("Inter", 14, "italic"))
        self.drop_label.grid(row=0, column=0, pady=20)

        # --- Right Side: Controls ---
        self.control_frame = ctk.CTkFrame(self.main_container, corner_radius=15)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Title
        ctk.CTkLabel(self.control_frame, text="Parameters", font=("Inter", 18, "bold")).pack(pady=(15, 10))

        # Prompt
        ctk.CTkLabel(self.control_frame, text="Prompt", font=("Inter", 12)).pack(anchor="w", padx=20)
        self.prompt_entry = ctk.CTkTextbox(self.control_frame, height=120)
        self.prompt_entry.pack(fill="x", padx=20, pady=(0, 15))
        self.prompt_entry.insert("0.0", "A cinematic video of the scene")

        # Resolution
        ctk.CTkLabel(self.control_frame, text="Resolution Preset", font=("Inter", 12)).pack(anchor="w", padx=20)
        self.size_var = ctk.StringVar(value=DEFAULT_SIZE)
        self.size_menu = ctk.CTkComboBox(self.control_frame, values=["480*832", "832*480", "576*1024", "1024*576", "Custom"], variable=self.size_var, command=self._on_resolution_change)
        self.size_menu.pack(fill="x", padx=20, pady=(0, 5))

        # Custom Resolution Fields (Initially hidden or integrated)
        self.custom_res_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.custom_res_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        self.width_var = ctk.StringVar(value="832")
        self.height_var = ctk.StringVar(value="480")
        
        ctk.CTkLabel(self.custom_res_frame, text="W:", font=("Inter", 11)).pack(side="left")
        self.width_entry = ctk.CTkEntry(self.custom_res_frame, width=60, textvariable=self.width_var)
        self.width_entry.pack(side="left", padx=5)
        
        ctk.CTkLabel(self.custom_res_frame, text="H:", font=("Inter", 11)).pack(side="left")
        self.height_entry = ctk.CTkEntry(self.custom_res_frame, width=60, textvariable=self.height_var)
        self.height_entry.pack(side="left", padx=5)
        
        self._on_resolution_change(DEFAULT_SIZE) # Initialize fields

        # Steps Slider
        ctk.CTkLabel(self.control_frame, text="Sampling Steps", font=("Inter", 12)).pack(anchor="w", padx=20)
        self.steps_slider = ctk.CTkSlider(self.control_frame, from_=10, to=100, number_of_steps=90)
        self.steps_slider.set(DEFAULT_STEPS)
        self.steps_slider.pack(fill="x", padx=20, pady=(0, 5))
        self.steps_label = ctk.CTkLabel(self.control_frame, text=str(DEFAULT_STEPS))
        self.steps_label.pack()
        self.steps_slider.configure(command=lambda v: self.steps_label.configure(text=str(int(v))))

        # Guidance Scale
        ctk.CTkLabel(self.control_frame, text="Guidance Scale", font=("Inter", 12)).pack(anchor="w", padx=20)
        self.guide_slider = ctk.CTkSlider(self.control_frame, from_=1.0, to=15.0, number_of_steps=140)
        self.guide_slider.set(DEFAULT_GUIDE)
        self.guide_slider.pack(fill="x", padx=20, pady=(0, 5))
        self.guide_label = ctk.CTkLabel(self.control_frame, text=str(DEFAULT_GUIDE))
        self.guide_label.pack()
        self.guide_slider.configure(command=lambda v: self.guide_label.configure(text=f"{v:.1f}"))

        # Frame Count
        ctk.CTkLabel(self.control_frame, text="Frame Count", font=("Inter", 12)).pack(anchor="w", padx=20)
        self.frames_entry = ctk.CTkEntry(self.control_frame)
        self.frames_entry.insert(0, str(DEFAULT_FRAMES))
        self.frames_entry.pack(fill="x", padx=20, pady=(0, 15))

        # Seed
        ctk.CTkLabel(self.control_frame, text="Seed (-1 for random)", font=("Inter", 12)).pack(anchor="w", padx=20)
        self.seed_entry = ctk.CTkEntry(self.control_frame)
        self.seed_entry.insert(0, "-1")
        self.seed_entry.pack(fill="x", padx=20, pady=(0, 15))

        # Camera Pose (Action Path)
        ctk.CTkLabel(self.control_frame, text="Camera Control (Action Path)", font=("Inter", 12)).pack(anchor="w", padx=20)
        self.action_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.action_frame.pack(fill="x", padx=20, pady=(0, 15))
        self.action_label = ctk.CTkLabel(self.action_frame, text="No path selected", font=("Inter", 11), anchor="w")
        self.action_label.pack(side="left", fill="x", expand=True)
        self.action_btn = ctk.CTkButton(self.action_frame, text="Select Folder", width=100, command=self._select_action_path)
        self.action_btn.pack(side="right")

        # VRAM Mode Toggle
        self.vram_mode_var = ctk.StringVar(value="low")
        self.vram_switch = ctk.CTkSwitch(self.control_frame, text="High VRAM Mode (48GB+)", variable=self.vram_mode_var, onvalue="high", offvalue="low")
        self.vram_switch.pack(pady=(0, 15))

        # --- Footer: Progress & Generate ---
        self.footer_frame = ctk.CTkFrame(self, height=180)
        self.footer_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.footer_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(self.footer_frame)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, padx=20, pady=15, sticky="ew")

        self.generate_btn = ctk.CTkButton(self.footer_frame, text="Start Video Generation", height=60, font=("Inter", 18, "bold"), command=self._start_generation)
        self.generate_btn.grid(row=1, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.footer_frame, text="Idle", font=("Inter", 13, "italic"))
        self.status_label.grid(row=2, column=0, pady=(0, 10))

    def log(self, message):
        self.status_label.configure(text=message)
        print(f"[GUI] {message}")

    def _select_input_image(self):
        """Open a file dialog to select the input image."""
        path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if path:
            self._load_image(path)

    def _load_image(self, path):
        """Process and preview the selected image."""
        try:
            self.image_path = path
            self.log(f"Image loaded: {os.path.basename(path)}")
            
            # Show preview
            img = Image.open(path)
            # Keep aspect ratio for preview
            w, h = img.size
            preview_size = 400
            if w > h:
                new_w = preview_size
                new_h = int(h * (preview_size / w))
            else:
                new_h = preview_size
                new_w = int(w * (preview_size / h))
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(new_w, new_h))
            self.drop_label.configure(image=ctk_img, text="")
        except Exception as e:
            self.log(f"Error loading image: {str(e)}")
            print(f"[ERROR] {str(e)}")

    def _select_action_path(self):
        path = filedialog.askdirectory(title="Select Camera Poses Folder")
        if path:
            # Simple validation
            if os.path.exists(os.path.join(path, "poses.npy")) or os.path.exists(os.path.join(path, "intrinsics.npy")):
                self.action_path = path
                self.action_label.configure(text=f"Selected: {os.path.basename(path)}")
            else:
                self.log("Warning: Folder doesn't contain poses.npy/intrinsics.npy")
                self.action_path = path
                self.action_label.configure(text=f"Selected (Incomplete): {os.path.basename(path)}")
    
    def _on_resolution_change(self, choice):
        if choice != "Custom" and "*" in choice:
            h, w = choice.split("*")
            self.height_var.set(h)
            self.width_var.set(w)
            self.width_entry.configure(state="disabled")
            self.height_entry.configure(state="disabled")
        else:
            self.width_entry.configure(state="normal")
            self.height_entry.configure(state="normal")

    def _validate_inputs(self):
        """Validate all UI inputs before starting generation."""
        try:
            # 1. Prompt
            prompt = self.prompt_entry.get("0.0", "end").strip()
            if not prompt:
                raise ValueError("Prompt cannot be empty.")

            # 2. Resolution (Width/Height)
            try:
                w = int(self.width_var.get())
                h = int(self.height_var.get())
                if w <= 0 or h <= 0:
                    raise ValueError("Width and Height must be positive integers.")
                # Recommend multiples of 16 for better VAE performance, though script handles rounding
                if w % 16 != 0 or h % 16 != 0:
                    self.log(f"Note: Resolution {w}x{h} will be rounded to nearest patch boundary.")
            except ValueError:
                raise ValueError("Width and Height must be valid integers.")

            # 3. Frame Count
            try:
                frames = int(self.frames_entry.get())
                if frames <= 0:
                    raise ValueError("Frame Count must be a positive integer.")
            except ValueError:
                raise ValueError("Frame Count must be a valid integer.")

            # 4. Seed
            try:
                seed = int(self.seed_entry.get())
                if seed < -1:
                    raise ValueError("Seed must be -1 or a positive integer.")
            except ValueError:
                raise ValueError("Seed must be a valid integer.")

            return True

        except ValueError as e:
            self.log(f"Validation Error: {str(e)}")
            return False

    def _start_generation(self):
        if not self.image_path:
            self.log("Error: Please select an image first.")
            return
        
        if self.is_generating:
            return

        if not self._validate_inputs():
            return

        self.is_generating = True
        self.generate_btn.configure(state="disabled", text="Generating...")
        self.progress_bar.set(0)
        
        # Run in separate thread
        threading.Thread(target=self._generation_worker, daemon=True).start()

    def _generation_worker(self):
        try:
            # 1. Initialize pipeline if needed
            vram_mode = self.vram_mode_var.get()
            
            if not self.pipeline or self.pipeline.vram_mode != vram_mode:
                self.log(f"Initializing models ({vram_mode} VRAM mode)...")
                # If pipeline exists but mode changed, we should probably clean up
                if self.pipeline:
                    self.pipeline = None
                    torch.cuda.empty_cache()
                
                # Use current directory as ckpt_dir
                self.pipeline = WanI2V_PreQuant(
                    checkpoint_dir=str(Path(__file__).parent),
                    vram_mode=vram_mode
                )
            
            # 2. Extract parameters
            prompt = self.prompt_entry.get("0.0", "end").strip()
            h = int(self.height_var.get())
            w = int(self.width_var.get())
            steps = int(self.steps_slider.get())
            guide = float(self.guide_slider.get())
            frames = int(self.frames_entry.get())
            seed = int(self.seed_entry.get())
            
            output_path = f"output_{int(time.time())}.mp4"
            
            # 3. Generate
            self.log(f"Generating video for: {os.path.basename(self.image_path)}")
            img = Image.open(self.image_path).convert("RGB")
            
            def progress_cb(current, total):
                progress = current / total
                self.after(0, lambda: self.progress_bar.set(progress))
                self.after(0, lambda: self.status_label.configure(text=f"Sampling: {current}/{total} steps"))

            video = self.pipeline.generate(
                input_prompt=prompt,
                img=img,
                action_path=self.action_path,
                max_area=h * w,
                frame_num=frames,
                sampling_steps=steps,
                guide_scale=guide,
                seed=seed,
                progress_callback=progress_cb
            )
            
            # 4. Save
            self.log("Saving video...")
            save_video(video, output_path)
            self.log(f"Done! Saved to {output_path}")
            
        except torch.cuda.OutOfMemoryError:
            self.log("Error: CUDA Out of Memory. Try lower resolution or 'Low VRAM Mode'.")
            if self.pipeline:
                self.pipeline = None # Free memory
                torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            msg = str(e) or type(e).__name__
            if "out of memory" in msg.lower():
                self.log("Error: CUDA Out of Memory. Try lower resolution.")
            else:
                self.log(f"Error: {msg}")
            print(tb)
            try:
                with open("crash_log.txt", "w") as f:
                    f.write(tb)
            except:
                pass
        finally:
            self.is_generating = False
            self.after(0, lambda: self.generate_btn.configure(state="normal", text="Start Video Generation"))
            self.after(0, lambda: self.progress_bar.set(0))

if __name__ == "__main__":
    app = App()
    app.mainloop()
