from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import time
import os
from pathlib import Path

from .localization import UI, LANG_EN, LANG_UK, pretty_class


class gui_functions:
    def __init__(self, model, ui):
        self.model = model
        self.ui = ui
        self.after_id = None
        self.lang = LANG_EN
        self.last_raw_classes = []
        self.cap = None
        self.is_playing = False
        self.base_delay_ms = 30
        self.speed = 1.0
        self.total_frames = 0
        self.fps = 30
        self.slider_block = False
        self.original_img = None 
        self.annotated_img = None 
        self.show_detection = True
        self.preprocess_enabled = True
        self.preprocess_mode = "clahe" 
        self.last_image_path = None
        self.last_image_bgr = None
        self.folder_path = None
        self.image_paths = []
        self.image_index = -1
        self.save_dir = None

        self.log("App started")

    # LANGUAGE

    def set_language(self, lang: str):
        if lang not in (LANG_EN, LANG_UK):
            lang = LANG_EN
        self.lang = lang
        self.ui.apply_language(lang)
        self._update_result_text_from_last()
        self.log(f"Language set: {lang}")

    def _S(self, key: str) -> str:
        return UI.get(self.lang, UI[LANG_EN]).get(key, UI[LANG_EN].get(key, key))

    # MINI LOG

    def log(self, msg: str):
        if not hasattr(self.ui, "log_box"):
            return
        try:
            t = time.strftime("%H:%M:%S")
            self.ui.log_box.configure(state="normal")
            self.ui.log_box.insert("end", f"[{t}] {msg}\n")
            self.ui.log_box.see("end")
            self.ui.log_box.configure(state="disabled")
        except Exception:
            pass

    # OpenCV PREPROCESS

    def preprocess_bgr(self, frame_bgr):
        if not self.preprocess_enabled or self.preprocess_mode == "none":
            return frame_bgr
        den = cv2.fastNlMeansDenoisingColored(frame_bgr, None, 3, 3, 7, 21)
        lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        return out

    def _bgr_to_pil_rgb(self, frame_bgr) -> Image.Image:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    # PREPROCESS SWITCH

    def set_preprocess(self, enabled: bool):
        self.preprocess_enabled = bool(enabled)
        state = "ON" if self.preprocess_enabled else "OFF"
        self.log(f"Preprocess set: {state}")

        if hasattr(self.ui, "drop_hint"):
            self.ui.drop_hint.configure(
                text="Preprocess: ON (CLAHE)" if self.preprocess_enabled else "Preprocess: OFF"
            )
        if (not self.is_playing) and (self.last_image_bgr is not None):
            try:
                self._run_detection_on_bgr(self.last_image_bgr)
            except Exception as e:
                self.log(f"Preprocess rerun error: {e}")

    # IMAGE

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp *.webp")])
        if not path:
            return
        try:
            self.last_image_path = path
            self.log(f"Open image: {os.path.basename(path)}")
            self._run_detection_on_image(path)
        except Exception as e:
            if hasattr(self.ui, "result_text"):
                self.ui.result_text.configure(text=f"Error opening image: {e}")
            self.log(f"Error open image: {e}")

    def _stop_video_if_running(self):
        self.is_playing = False
        if self.after_id is not None:
            try:
                self.ui.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _update_result_text_from_last(self):
        if not hasattr(self.ui, "result_text"):
            return
        if not self.last_raw_classes:
            self.ui.result_text.configure(text=self._S("no_detections_yet"))
            return
        uniq = sorted(set(self.last_raw_classes))
        nice = [pretty_class(x, self.lang) for x in uniq]
        self.ui.result_text.configure(text=self._S("detected_signs") + ", ".join(nice))

    def _run_detection_on_image(self, path: str):
        self._stop_video_if_running()

        frame_bgr = cv2.imread(path)
        if frame_bgr is None:
            raise RuntimeError("cv2.imread failed (unsupported file or path)")
        self.last_image_bgr = frame_bgr.copy()
        self.last_image_path = path

        self._run_detection_on_bgr(frame_bgr)

    def _run_detection_on_bgr(self, frame_bgr):
        self.original_img = self._bgr_to_pil_rgb(frame_bgr)
        frame_infer = self.preprocess_bgr(frame_bgr)

        t0 = time.perf_counter()
        t_infer0 = time.perf_counter()
        results = self.model(frame_infer)
        t_infer1 = time.perf_counter()
        t1 = time.perf_counter()

        annotated_bgr = results[0].plot()
        self.annotated_img = self._bgr_to_pil_rgb(annotated_bgr)

        infer_ms = (t_infer1 - t_infer0) * 1000.0
        fps_one = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0

        self.show_detection = True
        self.display_current_image()

        names = results[0].names
        total_objects = 0
        conf_values = []
        detected_raw = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            raw = names[cls_id]
            conf = float(box.conf[0])
            detected_raw.append(raw)
            conf_values.append(conf)
            total_objects += 1

        self.last_raw_classes = detected_raw[:]

        if conf_values:
            avg_conf = sum(conf_values) / len(conf_values)
            min_conf = min(conf_values)
            max_conf = max(conf_values)
            conf_text = f'{self._S("confidence")}: avg {avg_conf*100:.1f}% (min {min_conf*100:.1f}%, max {max_conf*100:.1f}%)'
        else:
            conf_text = f'{self._S("confidence")}: -'

        if hasattr(self.ui, "stats_total"):
            self.ui.stats_total.configure(text=f'{self._S("objects")}: {total_objects}')
        if hasattr(self.ui, "stats_fps"):
            self.ui.stats_fps.configure(text=f'{self._S("fps")}: {fps_one:.2f}')
        if hasattr(self.ui, "stats_infer"):
            self.ui.stats_infer.configure(text=f'{self._S("inference")}: {infer_ms:.1f} ms')
        if hasattr(self.ui, "stats_conf"):
            self.ui.stats_conf.configure(text=conf_text)

        if detected_raw:
            uniq = sorted(set(detected_raw))
            nice = [pretty_class(x, self.lang) for x in uniq]
            if hasattr(self.ui, "result_text"):
                self.ui.result_text.configure(text=self._S("detected_signs") + ", ".join(nice))
            self.log("Detected: " + ", ".join(nice))
        else:
            if hasattr(self.ui, "result_text"):
                self.ui.result_text.configure(text=self._S("no_detections"))
            self.log(self._S("no_detections"))

        if hasattr(self.ui, "drop_hint"):
            self.ui.drop_hint.configure(
                text="Preprocess: ON (CLAHE)" if self.preprocess_enabled else "Preprocess: OFF"
            )

    def toggle_detection(self):
        if self.original_img is None or self.annotated_img is None:
            return
        self.show_detection = not self.show_detection
        self.display_current_image()
        self.log(f"Toggle detection: {'ON' if self.show_detection else 'OFF'}")

    def display_current_image(self):
        img = self.annotated_img if self.show_detection else self.original_img
        img_copy = img.copy()
        img_copy.thumbnail((820, 560))
        photo = ImageTk.PhotoImage(img_copy, master=self.ui)
        self.ui.image_label.configure(image=photo, text="")
        self.ui.image_label.image = photo

    def clear_image(self):
        self._stop_video_if_running()
        self.original_img = None
        self.annotated_img = None
        self.last_raw_classes = []
        self.last_image_path = None
        self.last_image_bgr = None

        self.ui.image_label.configure(image="", text="Use buttons to open image")
        self.ui.image_label.image = None

        if hasattr(self.ui, "result_text"):
            self.ui.result_text.configure(text=self._S("no_detections_yet"))

        if hasattr(self.ui, "stats_total"):
            self.ui.stats_total.configure(text=f'{self._S("objects")}: -')
        if hasattr(self.ui, "stats_fps"):
            self.ui.stats_fps.configure(text=f'{self._S("fps")}: -')
        if hasattr(self.ui, "stats_infer"):
            self.ui.stats_infer.configure(text=f'{self._S("inference")}: - ms')
        if hasattr(self.ui, "stats_conf"):
            self.ui.stats_conf.configure(text=f'{self._S("confidence")}: -')

        if hasattr(self.ui, "time_label"):
            self.ui.time_label.configure(text="00:00 / 00:00")
        if hasattr(self.ui, "video_slider"):
            self.ui.video_slider.set(0)

        if hasattr(self.ui, "drop_hint"):
            self.ui.drop_hint.configure(text="")

        self.log("Cleared image")

    # FOLDER NAVIGATION

    def open_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        paths = []
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(exts):
                paths.append(os.path.join(folder, name))

        if not paths:
            if hasattr(self.ui, "result_text"):
                self.ui.result_text.configure(text="No images found in folder")
            self.log("No images found in selected folder")
            return

        self.folder_path = folder
        self.image_paths = paths
        self.image_index = 0

        self.log(self._S("folder_loaded").format(n=len(self.image_paths)))
        self._run_detection_on_image(self.image_paths[self.image_index])

    def next_image(self):
        if not self.image_paths:
            self.log("Next: folder not loaded")
            return
        self.image_index = min(len(self.image_paths) - 1, self.image_index + 1)
        self.log(f"Next ({self.image_index+1}/{len(self.image_paths)}): {os.path.basename(self.image_paths[self.image_index])}")
        self._run_detection_on_image(self.image_paths[self.image_index])

    def prev_image(self):
        if not self.image_paths:
            self.log("Prev: folder not loaded")
            return
        self.image_index = max(0, self.image_index - 1)
        self.log(f"Prev ({self.image_index+1}/{len(self.image_paths)}): {os.path.basename(self.image_paths[self.image_index])}")
        self._run_detection_on_image(self.image_paths[self.image_index])

    # BATCH

    def batch_process_folder(self):
        if not self.image_paths:
            self.open_folder()
            if not self.image_paths:
                return

        parent_out = filedialog.askdirectory(title="Select parent folder to save batch results")
        if not parent_out:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(parent_out, f"batch_results_{ts}")
        os.makedirs(results_dir, exist_ok=True)

        self.save_dir = results_dir
        self._stop_video_if_running()

        total = len(self.image_paths)
        self.log(f"Batch started: {total} images")
        self.log(f"Save to: {results_dir}")

        for i, p in enumerate(self.image_paths, start=1):
            try:
                self._run_detection_on_image(p)
                if self.annotated_img is not None:
                    base = Path(p).stem
                    save_path = os.path.join(results_dir, f"{base}_detected.jpg")
                    self.annotated_img.save(save_path, quality=95)

                if hasattr(self.ui, "result_text"):
                    self.ui.result_text.configure(text=f"Batch: {i}/{total} processed...")
                self.ui.update_idletasks()
            except Exception as e:
                self.log(f"Batch error on {os.path.basename(p)}: {e}")

        if hasattr(self.ui, "result_text"):
            self.ui.result_text.configure(text=self._S("batch_done").format(n=total, path=results_dir))
        self.log("Batch finished")

    def save_all_results(self):
        self.log("Save results: already saved during batch")
        if hasattr(self.ui, "result_text"):
            self.ui.result_text.configure(text="All detected images already saved during batch processing.")

    # VIDEO

    def open_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        self.log(f"Open video: {os.path.basename(path)}")
        self.open_video_path(path)

    def open_video_path(self, path: str):
        self.is_playing = False
        if self.after_id is not None:
            try:
                self.ui.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if hasattr(self.ui, "video_slider"):
            self.ui.video_slider.configure(to=self.total_frames)
            self.ui.video_slider.set(0)

        self.update_time_label(0)
        self.play_video()

    def play_video(self):
        if self.cap is None:
            return
        if self.is_playing:
            return
        self.is_playing = True
        self.log("Video: play")
        self._video_loop()

    def _video_loop(self):
        if not self.is_playing or self.cap is None:
            return

        ret, frame_bgr = self.cap.read()
        if not ret:
            self.is_playing = False
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            self.log("Video ended")
            return

        frame_infer = self.preprocess_bgr(frame_bgr)
        results = self.model(frame_infer)
        annotated_bgr = results[0].plot()

        img = self._bgr_to_pil_rgb(annotated_bgr)
        img.thumbnail((820, 560))
        photo = ImageTk.PhotoImage(img, master=self.ui)

        self.ui.image_label.configure(image=photo, text="")
        self.ui.image_label.image = photo

        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if hasattr(self.ui, "video_slider") and not self.slider_block:
            self.ui.video_slider.set(current_frame)

        self.update_time_label(current_frame)

        delay = max(1, int(self.base_delay_ms / self.speed))
        self.after_id = self.ui.after(delay, self._video_loop)

    def pause_video(self):
        self.is_playing = False
        self.log("Video: pause")

    def seek_video(self, seconds):
        if self.cap is None:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        current = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        new_frame = max(0, current + int(fps * seconds))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.log(f"Video seek: {seconds:+}s")

    def set_speed(self, speed: float):
        self.speed = max(0.25, min(4.0, float(speed)))
        if hasattr(self.ui, "status_text"):
            self.ui.status_text.configure(text=f"Speed: {self.speed}x")
        self.log(f"Speed set: {self.speed}x")

    def format_time(self, seconds: int) -> str:
        m = seconds // 60
        s = seconds % 60
        return f"{m:02d}:{s:02d}"

    def update_time_label(self, current_frame: int):
        current_sec = int(current_frame / self.fps) if self.fps else 0
        total_sec = int(self.total_frames / self.fps) if self.fps else 0
        if hasattr(self.ui, "time_label"):
            self.ui.time_label.configure(text=f"{self.format_time(current_sec)} / {self.format_time(total_sec)}")

    def on_slider_move(self, value):
        if self.cap is None:
            return
        self.slider_block = True
        frame = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self.update_time_label(frame)
        self.slider_block = False
