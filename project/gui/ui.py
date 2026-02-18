import os
import sys
import customtkinter as ctk
from PIL import Image, ImageTk

from .functiongui import gui_functions
from .localization import UI, LANG_EN, LANG_UK

BG_MAIN = "#1E1E2E"
BG_PANEL = "#2A2A3C"
ACCENT = "#3A86FF"
ACCENT_HOVER = "#265DCC"
TEXT = "#FFFFFF"

def _set_windows_taskbar_icon(root: ctk.CTk, ico_path: str, png_fallback_path: str | None = None):
    if sys.platform.startswith("win"):
        try:
            import ctypes
            app_id = "RecognitionSystem.App"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

    if ico_path and os.path.exists(ico_path):
        try:
            root.iconbitmap(ico_path)
        except Exception:
            pass

    if png_fallback_path and os.path.exists(png_fallback_path):
        try:
            img = Image.open(png_fallback_path)
            photo = ImageTk.PhotoImage(img)
            root.wm_iconphoto(True, photo)
            root._iconphoto_ref = photo
        except Exception:
            pass


class App(ctk.CTk):
    def __init__(self, model):
        super().__init__()

        self.title("Recognition System")
        self.geometry("1100x720")
        self.resizable(False, False)
        self.configure(fg_color=BG_MAIN)

        base_dir = os.path.dirname(__file__)
        ico_path = os.path.join(base_dir, "icon2.ico")
        png_path = os.path.join(base_dir, "icon2.png")
        _set_windows_taskbar_icon(self, ico_path=ico_path, png_fallback_path=png_path)

        self.controller = gui_functions(model, self)

        top_area = ctk.CTkFrame(self, fg_color="transparent")
        top_area.pack(fill="both", expand=True)

        bottom_area = ctk.CTkFrame(self, height=150, fg_color=BG_PANEL)
        bottom_area.pack(fill="x", padx=8, pady=(0, 8))
        bottom_area.pack_propagate(False)

        bottom_area.grid_columnconfigure(0, weight=3)
        bottom_area.grid_columnconfigure(1, weight=1)
        bottom_area.grid_rowconfigure(0, weight=1)

        bottom_left = ctk.CTkFrame(bottom_area, fg_color="transparent")
        bottom_left.grid(row=0, column=0, sticky="nsew", padx=(10, 8), pady=10)

        bottom_right = ctk.CTkFrame(bottom_area, fg_color="transparent")
        bottom_right.grid(row=0, column=1, sticky="nsew", padx=(8, 10), pady=10)

        left = ctk.CTkFrame(top_area, width=200, fg_color=BG_PANEL)
        left.pack(side="left", fill="y", padx=8, pady=8)
        left.pack_propagate(False)

        center = ctk.CTkFrame(top_area, fg_color=BG_PANEL)
        center.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        right = ctk.CTkFrame(top_area, width=220, fg_color=BG_PANEL)
        right.pack(side="right", fill="y", padx=8, pady=8)
        right.pack_propagate(False)

        def make_btn(parent, text, cmd):
            return ctk.CTkButton(
                parent, text=text, command=cmd,
                fg_color=ACCENT, hover_color=ACCENT_HOVER,
                text_color=TEXT, height=36
            )

        # LEFT
        self.lbl_images = ctk.CTkLabel(left, text="", font=("Arial", 16, "bold"), text_color=TEXT)
        self.lbl_images.pack(pady=(12, 6))

        self.lang_menu = ctk.CTkOptionMenu(
            left,
            values=[LANG_EN, LANG_UK],
            command=self.controller.set_language,
            fg_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            text_color=TEXT
        )
        self.lang_menu.pack(pady=(0, 10), padx=8, fill="x")
        self.lang_menu.set(LANG_EN)

        # Preprocess ON/OFF Switch
        self.preprocess_switch = ctk.CTkSwitch(
            left,
            text="Preprocess: ON",
            command=self._on_preprocess_toggle,
            fg_color=ACCENT,
            progress_color=ACCENT,
            text_color=TEXT
        )
        self.preprocess_switch.pack(pady=(0, 10), padx=8, fill="x")
        self.preprocess_switch.select() 
        self.controller.set_preprocess(True)

        self.btn_choose_image = make_btn(left, "", self.controller.open_image)
        self.btn_choose_image.pack(pady=8, fill="x", padx=8)

        self.btn_toggle = make_btn(left, "", self.controller.toggle_detection)
        self.btn_toggle.pack(pady=8, fill="x", padx=8)

        self.btn_clear = make_btn(left, "", self.controller.clear_image)
        self.btn_clear.pack(pady=8, fill="x", padx=8)

        self.btn_folder = make_btn(left, "", self.controller.open_folder)
        self.btn_folder.pack(pady=(8, 12), fill="x", padx=8)

        nav = ctk.CTkFrame(left, fg_color="transparent")
        nav.pack(pady=(10, 10))

        ctk.CTkButton(
            nav, text="⬅", command=self.controller.prev_image,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=TEXT, width=56, height=42
        ).pack(side="left", padx=12)

        ctk.CTkButton(
            nav, text="➡", command=self.controller.next_image,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color=TEXT, width=56, height=42
        ).pack(side="left", padx=12)

        self.btn_save = make_btn(left, "", self.controller.save_all_results)
        self.btn_save.pack(side="bottom", pady=(8, 12), fill="x", padx=8)

        self.btn_batch = make_btn(left, "", self.controller.batch_process_folder)
        self.btn_batch.pack(side="bottom", pady=(0, 8), fill="x", padx=8)

        # CENTER
        self.image_label = ctk.CTkLabel(center, text="Use buttons to open image", text_color=TEXT)
        self.image_label.pack(expand=True)

        self.drop_hint = ctk.CTkLabel(center, text="", text_color=TEXT)
        self.drop_hint.pack(pady=(0, 10))

        # RIGHT
        self.lbl_video = ctk.CTkLabel(right, text="", font=("Arial", 16, "bold"), text_color=TEXT)
        self.lbl_video.pack(pady=15)

        self.btn_choose_video = make_btn(right, "", self.controller.open_video)
        self.btn_choose_video.pack(pady=6, fill="x", padx=8)

        self.btn_play = make_btn(right, "", self.controller.play_video)
        self.btn_play.pack(pady=6, fill="x", padx=8)

        self.btn_pause = make_btn(right, "", self.controller.pause_video)
        self.btn_pause.pack(pady=6, fill="x", padx=8)

        self.btn_seek_plus = make_btn(right, "", lambda: self.controller.seek_video(10))
        self.btn_seek_plus.pack(pady=6, fill="x", padx=8)

        self.btn_seek_minus = make_btn(right, "", lambda: self.controller.seek_video(-10))
        self.btn_seek_minus.pack(pady=6, fill="x", padx=8)

        self.btn_speed2 = make_btn(right, "", lambda: self.controller.set_speed(2.0))
        self.btn_speed2.pack(pady=10, fill="x", padx=8)

        self.btn_speed1 = make_btn(right, "", lambda: self.controller.set_speed(1.0))
        self.btn_speed1.pack(pady=6, fill="x", padx=8)

        self.btn_speed05 = make_btn(right, "", lambda: self.controller.set_speed(0.5))
        self.btn_speed05.pack(pady=6, fill="x", padx=8)

        self.time_label = ctk.CTkLabel(right, text="00:00 / 00:00", text_color=TEXT, font=("Arial", 14, "bold"))
        self.time_label.pack(pady=(15, 5))

        self.video_slider = ctk.CTkSlider(right, from_=0, to=100, command=self.controller.on_slider_move)
        self.video_slider.pack(fill="x", padx=10, pady=5)
        self.video_slider.set(0)

        self.status_text = ctk.CTkLabel(right, text="Speed: 1.0x", text_color=TEXT, font=("Arial", 14, "bold"))
        self.status_text.pack(pady=15)

        # BOTTOM LEFT
        self.lbl_result = ctk.CTkLabel(bottom_left, text="", font=("Arial", 18, "bold"), text_color=TEXT)
        self.lbl_result.pack(anchor="w", pady=(0, 6))

        self.result_text = ctk.CTkLabel(
            bottom_left, text="", text_color=TEXT,
            font=("Arial", 14), anchor="w", justify="left"
        )
        self.result_text.pack(fill="x", pady=(0, 8))

        stats_frame = ctk.CTkFrame(bottom_left, fg_color="transparent")
        stats_frame.pack(fill="x")

        self.stats_total = ctk.CTkLabel(stats_frame, text="", text_color=TEXT, font=("Arial", 14, "bold"))
        self.stats_total.grid(row=0, column=0, sticky="w", padx=(0, 20))

        self.stats_fps = ctk.CTkLabel(stats_frame, text="", text_color=TEXT, font=("Arial", 14, "bold"))
        self.stats_fps.grid(row=0, column=1, sticky="w", padx=(0, 20))

        self.stats_infer = ctk.CTkLabel(stats_frame, text="", text_color=TEXT, font=("Arial", 14, "bold"))
        self.stats_infer.grid(row=0, column=2, sticky="w", padx=(0, 20))

        self.stats_conf = ctk.CTkLabel(stats_frame, text="", text_color=TEXT, font=("Arial", 14, "bold"))
        self.stats_conf.grid(row=0, column=3, sticky="w", padx=(0, 20))

        # BOTTOM RIGHT
        self.lbl_log = ctk.CTkLabel(bottom_right, text="", font=("Arial", 16, "bold"), text_color=TEXT)
        self.lbl_log.pack(anchor="w", pady=(0, 6))

        self.log_box = ctk.CTkTextbox(bottom_right, height=105, fg_color=BG_MAIN, text_color=TEXT, corner_radius=10)
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

        self.apply_language(LANG_EN)

    #Switch callback
    
    def _on_preprocess_toggle(self):
        enabled = bool(self.preprocess_switch.get())
        self.preprocess_switch.configure(text=f"Preprocess: {'ON' if enabled else 'OFF'}")
        self.controller.set_preprocess(enabled)

    def apply_language(self, lang: str):
        s = UI.get(lang, UI[LANG_EN])

        self.lbl_images.configure(text=s["images_controls"])
        self.lbl_video.configure(text=s["video_controls"])

        self.btn_choose_image.configure(text=s["choose_image"])
        self.btn_toggle.configure(text=s["toggle_detection"])
        self.btn_clear.configure(text=s["clear_image"])
        self.btn_folder.configure(text=s["choose_folder"])

        self.btn_batch.configure(text=s["batch"])
        self.btn_save.configure(text=s["save"])

        self.btn_choose_video.configure(text=s["choose_video"])
        self.btn_play.configure(text=s["play"])
        self.btn_pause.configure(text=s["pause"])
        self.btn_seek_plus.configure(text=s["seek_plus"])
        self.btn_seek_minus.configure(text=s["seek_minus"])
        self.btn_speed2.configure(text=s["speed2"])
        self.btn_speed1.configure(text=s["speed1"])
        self.btn_speed05.configure(text=s["speed05"])

        self.lbl_result.configure(text=s["detection_result"])
        self.lbl_log.configure(text=s["log"])

        if not self.result_text.cget("text"):
            self.result_text.configure(text=s["no_detections_yet"])

        self.stats_total.configure(text=f'{s["objects"]}: -')
        self.stats_fps.configure(text=f'{s["fps"]}: -')
        self.stats_infer.configure(text=f'{s["inference"]}: - ms')
        self.stats_conf.configure(text=f'{s["confidence"]}: -')
