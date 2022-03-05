import tkinter as tk
from tkinter import PhotoImage, Button, ttk
import os
import cv2 as cv
import time
from tkinter.filedialog import askopenfile


root = tk.Tk()


def upload_file():
    file_path = askopenfile(mode='r', filetypes=[('Image Files', '*jpeg')])
    if file_path is not None:
        pass
    print("Selected:", file_path)
    output = tk.Label(root, text=file_path.name)
    output.grid()


def frontend():

    root.title("Diabetic Retinopathy Detection")

    window_width = 700
    window_height = 550
    # get the screen dimension
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # find the center point
    center_x = int(screen_width/2 - window_width / 2)
    center_y = int(screen_height/2 - window_height / 2)

    # set the position of the window to the center of the screen
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    file_upload = ttk.Button(root, text="Upload 2D Fundus Image",
                             command=lambda: upload_file())
    file_upload.grid()

    root.resizable(False, False)

    # when loading a file in, root.lower()

    root.mainloop()


frontend()
