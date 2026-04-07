"""
_installer_gui.py  --  Dynamo Magician graphical installer.

Called by setup.bat after Python/venv/packages are ready.
Shows each model file, lets the user click Download, and
enables Launch when everything is in place.
"""

import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

HERE = Path(__file__).parent
REPO = "cahlen/lingbot-world-base-cam-nf4"

# (repo filename, display label, approx size)
FILES = [
    ("models_t5_umt5-xxl-enc-bf16.pth",                "T5 text encoder",        "9.6 GB"),
    ("Wan2.1_VAE.pth",                                  "VAE",                    "1.6 GB"),
    ("high_noise_model_bnb_nf4/model.safetensors",      "High-noise model",       "9.5 GB"),
    ("high_noise_model_bnb_nf4/config.json",            "High-noise config",      "—"),
    ("high_noise_model_bnb_nf4/quantization_meta.json", "High-noise quant meta",  "—"),
    ("low_noise_model_bnb_nf4/model.safetensors",       "Low-noise model",        "9.5 GB"),
    ("low_noise_model_bnb_nf4/config.json",             "Low-noise config",       "—"),
    ("low_noise_model_bnb_nf4/quantization_meta.json",  "Low-noise quant meta",   "—"),
    ("tokenizer/tokenizer.json",                         "Tokenizer",              "—"),
    ("tokenizer/tokenizer_config.json",                  "Tokenizer config",       "—"),
    ("tokenizer/special_tokens_map.json",                "Special tokens map",     "—"),
]

KEY_FILES = [
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.1_VAE.pth",
    "high_noise_model_bnb_nf4/model.safetensors",
    "low_noise_model_bnb_nf4/model.safetensors",
    "tokenizer/tokenizer.json",
]

BG          = "#12121f"
BG_CARD     = "#1c1c30"
BG_CARD2    = "#16162a"
FG          = "#e0e0f0"
FG_DIM      = "#666688"
GREEN       = "#00e5a0"
YELLOW      = "#ffcc44"
RED         = "#ff4466"
BLUE        = "#4488ff"
FONT        = "Segoe UI"


def all_present() -> bool:
    return all((HERE / f).exists() for f in KEY_FILES)


class InstallerApp:
    def __init__(self) -> None:
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._cancel = threading.Event()
        self._downloading = False

        self.root = tk.Tk()
        self.root.title("Dynamo Magician — Setup")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self._build()
        self._center()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build(self) -> None:
        root = self.root
        PX = 24  # horizontal padding

        # ── header ──
        hdr = tk.Frame(root, bg=BG)
        hdr.pack(fill="x", padx=PX, pady=(24, 4))
        tk.Label(hdr, text="DYNAMO MAGICIAN",
                 font=(FONT, 22, "bold"), fg=GREEN, bg=BG).pack(anchor="w")
        tk.Label(hdr, text="First-Time Setup  ·  AI Model Download",
                 font=(FONT, 10), fg=FG_DIM, bg=BG).pack(anchor="w")

        sep = tk.Frame(root, height=1, bg=BG_CARD)
        sep.pack(fill="x", padx=PX, pady=(10, 14))

        # ── file list ──
        card = tk.Frame(root, bg=BG_CARD, bd=0)
        card.pack(fill="x", padx=PX, pady=(0, 12))

        inner = tk.Frame(card, bg=BG_CARD)
        inner.pack(fill="x", padx=12, pady=8)

        self._rows: dict[str, dict] = {}
        for filename, label, size in FILES:
            present = (HERE / filename).exists()
            row = tk.Frame(inner, bg=BG_CARD2, pady=3)
            row.pack(fill="x", pady=2)

            icon_var = tk.StringVar(value="✓" if present else "·")
            icon_lbl = tk.Label(row, textvariable=icon_var, width=2,
                                font=(FONT, 11, "bold"),
                                fg=GREEN if present else FG_DIM,
                                bg=BG_CARD2)
            icon_lbl.pack(side="left", padx=(8, 4))

            tk.Label(row, text=label, font=(FONT, 10), fg=FG,
                     bg=BG_CARD2, anchor="w", width=24).pack(side="left")

            status_var = tk.StringVar(
                value=("already downloaded" if present else size))
            status_lbl = tk.Label(row, textvariable=status_var,
                                  font=(FONT, 9), fg=FG_DIM if present else FG_DIM,
                                  bg=BG_CARD2, width=18, anchor="e")
            status_lbl.pack(side="right", padx=(4, 10))

            self._rows[filename] = {
                "icon_var":   icon_var,
                "status_var": status_var,
                "icon_lbl":   icon_lbl,
                "status_lbl": status_lbl,
            }

        # ── progress bar ──
        pb_frame = tk.Frame(root, bg=BG)
        pb_frame.pack(fill="x", padx=PX, pady=(0, 2))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("DM.Horizontal.TProgressbar",
                        troughcolor=BG_CARD, background=GREEN,
                        thickness=6)
        self._pb = ttk.Progressbar(pb_frame,
                                   style="DM.Horizontal.TProgressbar",
                                   mode="indeterminate", length=512)
        self._pb.pack(fill="x")

        self._current_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self._current_var,
                 font=(FONT, 9), fg=FG_DIM, bg=BG,
                 anchor="w").pack(fill="x", padx=PX, pady=(2, 10))

        # ── status message ──
        self._msg_var = tk.StringVar(
            value=("All files are ready — click Launch to start the app."
                   if all_present()
                   else "Click  Download AI Models  to fetch ~30 GB of model files.\n"
                        "This only happens once.  You can pause and resume at any time."))
        tk.Label(root, textvariable=self._msg_var,
                 font=(FONT, 10), fg=FG, bg=BG,
                 justify="left", wraplength=512,
                 anchor="w").pack(fill="x", padx=PX, pady=(0, 16))

        # ── buttons ──
        btn_frame = tk.Frame(root, bg=BG)
        btn_frame.pack(pady=(0, 24))

        self._dl_btn = tk.Button(
            btn_frame,
            text="  ⬇   Download AI Models  ",
            font=(FONT, 12, "bold"),
            fg=BG, bg=GREEN,
            activeforeground=BG, activebackground="#00bb88",
            relief="flat", padx=18, pady=10, cursor="hand2",
            command=self._on_download,
            state="disabled" if all_present() else "normal",
        )
        self._dl_btn.pack(side="left", padx=8)

        self._cancel_btn = tk.Button(
            btn_frame,
            text="  ✕  Cancel  ",
            font=(FONT, 11),
            fg=FG_DIM, bg=BG_CARD,
            activeforeground=FG, activebackground=BG_CARD2,
            relief="flat", padx=10, pady=10, cursor="hand2",
            command=self._on_cancel,
            state="disabled",
        )
        self._cancel_btn.pack(side="left", padx=4)

        self._launch_btn = tk.Button(
            btn_frame,
            text="  ▶   Launch App  ",
            font=(FONT, 12, "bold"),
            fg="#ffffff", bg=BLUE,
            activeforeground="#ffffff", activebackground="#2266cc",
            relief="flat", padx=18, pady=10, cursor="hand2",
            command=self._on_launch,
            state="normal" if all_present() else "disabled",
        )
        self._launch_btn.pack(side="left", padx=8)

    def _center(self) -> None:
        self.root.update_idletasks()
        w = self.root.winfo_reqwidth()
        h = self.root.winfo_reqheight()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 3
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    # ── button handlers ──────────────────────────────────────────────────────

    def _on_download(self) -> None:
        self._downloading = True
        self._cancel.clear()
        self._dl_btn.config(state="disabled",
                            text="  ⏳  Downloading…  ",
                            bg=BG_CARD, fg=FG_DIM)
        self._cancel_btn.config(state="normal")
        self._launch_btn.config(state="disabled")
        self._pb.config(mode="indeterminate")
        self._pb.start(14)
        self._msg_var.set("Downloading model files…  Do not close this window.")
        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()
        self._poll()

    def _on_cancel(self) -> None:
        self._cancel.set()
        self._cancel_btn.config(state="disabled")
        self._msg_var.set("Cancelling after current file finishes…")

    def _on_launch(self) -> None:
        self.root.destroy()
        run_bat = HERE / "run.bat"
        if run_bat.exists():
            os.startfile(str(run_bat))
        else:
            import subprocess
            subprocess.Popen(
                [sys.executable, "server.py"],
                cwd=str(HERE),
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )

    # ── download worker (background thread) ──────────────────────────────────

    def _worker(self) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            self._q.put(("error", "huggingface_hub is not installed.\nClose this window and run setup.bat again."))
            return

        total = len(FILES)
        for idx, (filename, label, size) in enumerate(FILES):
            if self._cancel.is_set():
                self._q.put(("cancelled",))
                return

            if (HERE / filename).exists():
                self._q.put(("skip", filename))
                continue

            self._q.put(("start", filename, label, size, idx, total))
            try:
                hf_hub_download(
                    repo_id=REPO,
                    filename=filename,
                    local_dir=str(HERE),
                    force_download=False,
                )
                self._q.put(("done", filename))
            except Exception as exc:
                self._q.put(("error", str(exc)))
                return

        self._q.put(("all_done",))

    # ── queue poller (main thread) ────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]

                if kind == "start":
                    _, filename, label, size, idx, total = msg
                    self._current_var.set(
                        f"  Downloading  ({idx + 1}/{total})  {label}"
                        + (f"  ·  {size}" if size != "—" else ""))
                    row = self._rows[filename]
                    row["icon_var"].set("⟳")
                    row["icon_lbl"].config(fg=YELLOW)
                    row["status_var"].set("downloading…")
                    row["status_lbl"].config(fg=YELLOW)

                elif kind in ("done", "skip"):
                    filename = msg[1]
                    row = self._rows[filename]
                    row["icon_var"].set("✓")
                    row["icon_lbl"].config(fg=GREEN)
                    row["status_var"].set("done" if kind == "done" else "already downloaded")
                    row["status_lbl"].config(fg=GREEN if kind == "done" else FG_DIM)

                elif kind == "all_done":
                    self._pb.stop()
                    self._pb.config(mode="determinate", value=100)
                    self._current_var.set("")
                    self._cancel_btn.config(state="disabled")
                    self._dl_btn.config(text="  ✓  All Files Ready  ",
                                        bg=BG_CARD, fg=GREEN)
                    self._launch_btn.config(state="normal")
                    self._msg_var.set("Download complete!  Click  Launch App  to start.")
                    self._downloading = False
                    return

                elif kind == "cancelled":
                    self._pb.stop()
                    self._current_var.set("")
                    self._cancel_btn.config(state="disabled")
                    self._dl_btn.config(state="normal",
                                        text="  ⬇   Download AI Models  ",
                                        bg=GREEN, fg=BG)
                    self._msg_var.set("Download paused.  Click Download to resume.")
                    self._downloading = False
                    return

                elif kind == "error":
                    self._pb.stop()
                    self._current_var.set("")
                    self._cancel_btn.config(state="disabled")
                    self._dl_btn.config(text="  ✗  Error — try again  ",
                                        bg=RED, fg="#ffffff", state="normal")
                    self._msg_var.set(f"Error: {msg[1]}")
                    self._downloading = False
                    return

        except queue.Empty:
            pass

        if self._downloading:
            self.root.after(150, self._poll)

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    InstallerApp().run()
