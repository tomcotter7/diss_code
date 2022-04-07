import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile
from PIL import ImageTk, Image
from utils.explainbility_utils import gradCAMplusplus, overlap_heatmap
import imghdr
import cv2 as cv
import pathlib
import tkinter.font as tkFont


# Python class to hold the Tkinter window and let the user interact with it.
class App:

    def __init__(self, model):

        self.root = tk.Tk()
        self.root.title("Diabetic Retinopathy Detection")
        self.root.config(bg="skyblue")
        self.mutli = False
        self.image = ""
        self.image_path = ""
        self.hm = ""
        self.model = model
        self.style = ttk.Style()
        self.style.configure('my.TButton', font=('Arial', 20),
                             background="grey", foreground="black")
        self.customFont = tkFont.Font(family="Arial", size=16)
        self.customBoldFont = tkFont.Font(family="Arial", size=16, weight=tkFont.BOLD)

        window_width = 1500
        window_height = 800
        # get the screen dimension
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # find the center point
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)

        # set the position of the window to the center of the screen
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        self.left_frame = tk.Frame(self.root, width=450, height=800, bg="grey")
        self.left_frame.pack(side="left", fill="both", padx=10, pady=10, expand=True)

        self.right_frame = tk.Frame(self.root, width=400, height=800, bg="grey")
        self.right_frame.pack(side="right", fill="both", padx=10, pady=10, expand=True)

        task_manager = tk.Frame(self.left_frame, width=400, height=300, bg="lightgrey")
        task_manager.pack(fill="both", padx=5, pady=5)

        self.image_frame = tk.Frame(self.left_frame, width=400,
                                    height=500, bg="grey")
        self.image_frame.pack(fill="both", padx=5, pady=5)

        file_upload = ttk.Button(task_manager, text="Upload Single 2D Fundus Image",
                                 command=lambda: self.upload_file(), style='my.TButton')
        file_upload.pack(fill="both", padx=20, pady=5)

        # multi_image = ttk.Button(task_manager, text="Upload Multiple 2d Fundus Images",
        #                         command=lambda: self.multi_image(), style='my.TButton')
        # multi_image.pack(fill="both", padx=20, pady=5)

        run_gpp = ttk.Button(task_manager, text="Run AI",
                             command=lambda: self.run_gpp(), style='my.TButton')
        run_gpp.pack(fill="both", padx=20, pady=5)

        self.root.resizable(False, False)

    def run(self):
        self.root.mainloop()

    def multi_image(self):
        # basically, import all the images into an array, and when run ai is clicked, run them all at once.
        # produce outputs but don't show heatmaps, only the output score, with a view heatmap button ?
        # we could actually show a progress bar in this case?
        self.multi = True
        pass

    def file_correct(self, file_path):
        if file_path is None:
            return False
        elif imghdr.what(file_path.name) is None:
            print("imghdr failed")
            return False

        return True

    # function to allow the user to open a file
    def upload_file(self):
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        file = askopenfile(mode='r', filetypes=[
                           ('Image Files', '*.jpeg'), ('All files', '*.*')])
        if not self.file_correct(file):
            output = tk.Label(self.image_frame, text="Please upload a image",
                              bg="red", font=self.customFont)
        else:
            self.image_path = file.name
            cv_img = cv.cvtColor(cv.imread(self.image_path), cv.COLOR_BGR2RGB)
            self.image = ImageTk.PhotoImage(image=Image.fromarray(cv_img))

            output = tk.Label(self.image_frame, text=file, image=self.image)

        file_name = tk.Label(self.image_frame, text=self.image_path, font=self.customFont)
        file_name.pack(fill="both", padx=5, pady=5)
        output.pack(fill="both", padx=5, pady=5)

    # function to return a label based on the class predicted by the model

    def getCorrectDR(self, cls):
        if cls == 0:
            return tk.Label(self.right_frame,
                            text="This eye is unlikely to have symptoms of D.R",
                            bg="green", font=self.customBoldFont)
        elif cls == 1:
            return tk.Label(self.right_frame,
                            text="This eye has symptoms of non-profilerative D.R",
                            bg="orange", font=self.customBoldFont)
        elif cls == 2:
            return tk.Label(self.right_frame,
                            text="This eye has symptoms of profilerative D.R, take immediate action",
                            bg="red", font=self.customBoldFont)
        else:
            return tk.Label(self.right_frame, text="error!", font=self.customFont)

    # function to allow the user to save the output images
    def save_images(self):

        path = pathlib.PurePath(self.image_path)
        parent_path = path.parents[1]
        name = str(path.name)
        folder = parent_path.joinpath('outputs')

        new_name_box = folder / (name.split(".")[0] + "_box.jpg")
        new_name_hm = folder / (name.split(".")[0] + "_hm.jpg")
        print(new_name_box)
        print(type(new_name_hm))

        dir_label = tk.Label(
            self.right_frame, text="images have been saved to " + str(folder), font=self.customFont)
        dir_label.pack(fill="both", padx=5, pady=5)

        ImageTk.getimage(self.image_box).convert('RGB').save(str(new_name_box), "JPEG")
        ImageTk.getimage(self.image_hm).convert('RGB').save(str(new_name_hm), "JPEG")

    # function to run the gradcam++ algorithm on an input image

    def run_gpp(self):
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        file_name = tk.Label(self.right_frame, text=self.image_path, font=self.customFont)
        file_name.pack(fill="both", padx=5, pady=5)
        if self.image != "":
            heatmap, output = gradCAMplusplus(
                self.image_path, self.model.model, self.model.last_conv)
            if heatmap is not None:
                overlayed, boxes = overlap_heatmap(self.image_path, heatmap, 0.3)

                self.image_hm = ImageTk.PhotoImage(overlayed)
                self.image_box = ImageTk.PhotoImage(boxes)
                output_text = tk.Label(
                    self.right_frame, text="Areas of caution", font=self.customBoldFont)
                output_text.pack(fill="both", padx=5, pady=5)
                gpp_output = tk.Label(
                    self.right_frame, text="output image", image=self.image_hm)
                gpp_output.pack(fill="both", padx=5, pady=5)
                boxes = tk.Label(self.right_frame, text="output image 2",
                                 image=self.image_box)
                boxes.pack(fill="both", padx=5, pady=5)
                save_button = ttk.Button(
                    self.right_frame, text="Save Output",
                    command=lambda: self.save_images(), style='my.TButton')
                save_button.pack(fill="both", padx=5, pady=5)
            text_output = self.getCorrectDR(output)
            text_output.pack(fill="both", padx=5, pady=5)

        else:
            error = tk.Label(self.right_frame,
                             text="Please select an image before trying anything",
                             bg="red", font=self.customFont)
            error.pack(fill="both", padx=5, pady=5)
