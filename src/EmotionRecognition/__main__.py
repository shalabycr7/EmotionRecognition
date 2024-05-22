import importlib
import os
import sys
from tkinter import filedialog

import ttkbootstrap as ttk
from keras.src.models.model import model_from_json
from keras.src.utils import img_to_array

from EmotionRecognition import DATADIR
from EmotionRecognition import DATADIR2

from PIL import Image, ImageTk

import cv2
import numpy as np


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        base_path = DATADIR

    return os.path.join(base_path, relative_path)


def resource_path2(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        base_path = DATADIR2

    return os.path.join(base_path, relative_path)


class MainApp(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Initialize variables
        self.dark_mode_state = True
        self.current_style = ttk.Style()

        # Store the app icons
        self.images = {}
        icon_names = ['import-file', 'import-file-dark', ]
        for name in icon_names:
            self.images[name] = ttk.PhotoImage(name=name, file=resource_path(f'{name}-icon.png'))
        # Create and pack widgets
        self.pack(fill='both', expand=True)
        self.create_main_gui()

    def create_main_gui(self):
        self.create_header()

        # Create container frame
        container = ttk.Frame(self, )
        container.pack(fill="both", expand=True)

        # Create grid and tools frames
        original_image_frame = ttk.Frame(container, padding=5, )
        detection_frame = ttk.Frame(container, padding=5, )

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

        self.import_btn = ttk.Button(
            master=header_frame,
            image=import_image,
            compound='left',
            style="link.TButton",
            command=self.open_file
        )
        self.import_btn.pack(side='right')

        ttk.Separator(header_frame, orient='vertical').pack(side='right', padx=20)

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

    # The process_image function integrates with the Tkinter app
    def process_image(self, file_path):
        # Load the model and Haar cascade file
        model = model_from_json(open(resource_path2('model.json'), "r").read())
        model.load_weights(resource_path2('model.h5'))
        face_haar_cascade = cv2.CascadeClassifier(resource_path2('haarcascade_frontalface_default.xml'))

        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            print("Error: Image not found. Please check the image path.")
            return

        def preprocess_image(image):
            resized_image = cv2.resize(image, (48, 48))  # Resize to model input size
            image_pixels = img_to_array(resized_image)  # Convert to NumPy array
            image_pixels = np.expand_dims(image_pixels, axis=0)  # Add a new dimension
            image_pixels /= 255.0  # Normalize to 0-1 range
            return image_pixels

        # Function to predict emotion
        def predict_emotion(model, image_pixels):
            predictions = model.predict(image_pixels)  # Make predictions
            max_index = np.argmax(predictions[0])  # Find class with highest probability
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            return emotion_prediction

        # Convert the image to grayscale and detect faces
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        # Process and draw results on the image
        for (x, y, w, h) in faces:
            cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y - 5:y + h + 5, x - 5:x + w + 5]
            roi_image = preprocess_image(roi_gray)

            emotion = predict_emotion(model, roi_image)

            # Add emotion text to the image
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 0.8
            FONT_THICKNESS = 2
            label_color = (10, 10, 255)
            text_x = x
            text_y = y - 10
            cv2.putText(image, emotion, (text_x, text_y), FONT, FONT_SCALE, label_color, FONT_THICKNESS)

        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the processed image to PIL Image format
        img_pil = Image.fromarray(image_rgb)

        # Convert the PIL Image to ImageTk format
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update the Tkinter label with the processed image
        self.detect_image_label.config(image=img_tk)
        self.detect_image_label.image = img_tk


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
