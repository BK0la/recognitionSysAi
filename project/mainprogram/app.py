import customtkinter as ctk
from ultralytics import YOLO
from mainprogram.modelload import load_model
from gui.ui import App

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

model = load_model()
app = App(model)
app.mainloop()