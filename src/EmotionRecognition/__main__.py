import importlib
import os
import re
import sqlite3
import sys
import threading
import time
from tkinter import messagebox, Listbox, filedialog

import ttkbootstrap as ttk
from ttkbootstrap import Toplevel
from ttkbootstrap.scrolled import ScrolledText, ScrolledFrame
from ttkbootstrap.toast import ToastNotification
from ttkbootstrap.tooltip import ToolTip

from EmotionRecognition import DATADIR
from PIL import Image, ImageTk
import numpy as np
import cv2


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        base_path = DATADIR

    return os.path.join(base_path, relative_path)


class MainApp(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Initialize variables
        self.dark_mode_state = True
        self.current_style = ttk.Style()

        # Store the app icons
        self.images = {}
        icon_names = ['import-file', 'import-file-dark', 'theme-toggle', 'theme-toggle-dark', ]
        for name in icon_names:
            self.images[name] = ttk.PhotoImage(name=name, file=resource_path(f'{name}-icon.png'))
        # Create and pack widgets
        self.pack(fill='both', expand=True)
        self.create_main_gui()
        self.set_theme()

    def set_theme(self):
        # Toggle dark mode state
        self.dark_mode_state = not self.dark_mode_state

        # Get current images for buttons
        toggle_img_key = 'theme-toggle' if self.dark_mode_state else 'theme-toggle-dark'
        import_img_key = 'import-file-dark' if self.dark_mode_state else 'import-file'

        current_images = {
            'toggle': self.images[toggle_img_key],
            'import': self.images[import_img_key],

        }

        # Configure style object
        theme_name = 'cyborg' if self.dark_mode_state else 'cosmo'
        self.current_style.theme_use(theme_name)

        # Configure button images
        self.theme_btn.configure(image=current_images['toggle'])
        self.import_btn.configure(image=current_images['import'])

        # Configure font
        font_config = "-family PlusJakartaSans -size 9"
        label_font_config = {'font': font_config}
        self.current_style.configure('TLabel', **label_font_config)

    def create_main_gui(self):
        self.create_header()

        # Create container frame
        container = ttk.Frame(self, )
        container.pack(fill="both", expand=True)

        # Create grid and tools frames
        original_image_frame = ttk.Frame(container, padding=5, bootstyle='info')
        detection_frame = ttk.Frame(container, padding=5, bootstyle='danger')

        # Place the frames in the grid, spanning 3 rows each
        original_image_frame.grid(row=0, column=0, sticky="nsew")
        detection_frame.grid(row=0, column=1, sticky="nsew")

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        # Create a label to display the image
        self.image_label = ttk.Label(original_image_frame)
        self.image_label.pack()
        # Create a label to display the image
        self.detect_image_label = ttk.Label(detection_frame)
        self.detect_image_label.pack()

    def create_header(self):
        # Create header
        header_frame = ttk.Frame(self, padding=(5, 2))
        header_frame.pack(fill='x', )

        ttk.Label(header_frame, text='Emotion Recognition', font="-family Barlow -size 13").pack(side='left')

        # Create import button
        import_image = self.images['import-file']
        theme_image = self.images['theme-toggle']

        self.import_btn = ttk.Button(
            master=header_frame,
            image=import_image,
            compound='left',
            style="link.TButton",
            command=self.open_file
        )
        self.import_btn.pack(side='right')

        # Create theme toggle button
        self.theme_btn = ttk.Button(
            master=header_frame,
            image=theme_image,
            style="link.TButton",
            command=self.set_theme
        )
        self.theme_btn.pack(side='right', padx=10)

        ttk.Separator(header_frame, orient='vertical').pack(side='right', padx=20)

        ToolTip(self.import_btn, delay=500, text="Clipboard History", bootstyle='primary')
        ToolTip(self.theme_btn, delay=500, text="Dark Mode", bootstyle='primary')

    def open_file(self):
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        if file_path:
            # Load the image
            image = Image.open(file_path)

            # Display the image in the Tkinter application
            img_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk  # Keep a reference

            # Process the image and predict emotion
            self.process_image(file_path)

    def process_image(self, file_path):
        # Load the image using OpenCV
        image = cv2.imread(file_path)

        # Convert the image to the format your model expects
        # (e.g. convert BGR to RGB and resize the image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224))  # Adjust size as needed

        # Predict the emotion using your AI model
        # emotion = predict_emotion(model, image_resized)

        # Display the emotion label in the application
        # emotion_label.config(text=f"Detected Emotion: {emotion}")

        # Optionally, you can also display the labeled image (e.g., with a bounding box and label)
        # Convert the image to a format that can be displayed using Tkinter
        labeled_image = Image.fromarray(image_rgb)

        # Display the labeled image (uncomment and adjust the following lines if needed)
        img_tk = ImageTk.PhotoImage(labeled_image)
        self.detect_image_label.config(image=img_tk)
        self.detect_image_label.image = img_tk  # Keep a reference


if __name__ == '__main__':
    # load a splash screen
    if '_PYIBoot_SPLASH' in os.environ and importlib.util.find_spec("pyi_splash"):
        import pyi_splash

        pyi_splash.update_text('UI Loaded ...')
        pyi_splash.close()

    window_width = 1500
    window_height = 800
    app = ttk.Window(title='EmotionRecognition',
                     size=[window_width, window_height], iconphoto=resource_path('favicon.png'))
    app.place_window_center()

    MainApp(app)
    app.mainloop()
