import tkinter as tk
from tkinter import PhotoImage, Button, ttk
import os
import cv2 as cv
import time
from tkinter.filedialog import askopenfile


class App:

    def __init__(self):

        self.root = tk.Tk()
        self.root.title("Diabetic Retinopathy Detection")
        self.image = ""

        window_width = 700
        window_height = 550
        # get the screen dimension
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # find the center point
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)

        # set the position of the window to the center of the screen
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        file_upload = ttk.Button(self.root, text="Upload 2D Fundus Image",
                                 command=lambda: self.upload_file())
        file_upload.grid()

        self.root.resizable(False, False)

        # when loading a file in, root.lower()

    def run(self):
        self.root.mainloop()

    def upload_file(self):
        file_path = askopenfile(mode='r', filetypes=[('Image Files', '*jpeg')])
        if file_path is not None:
            pass
        print("Selected:", file_path)
        output = tk.Label(self.root, text=file_path.name)
        output.grid()
        self.image = file_path.name


new_app = App()
new_app.run()
