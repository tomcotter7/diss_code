import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile
from PIL import ImageTk, Image
from utils.explainbility_utils import gradCAMplusplus, overlap_heatmap


class App:

    def __init__(self, model):

        self.root = tk.Tk()
        self.root.title("Diabetic Retinopathy Detection")
        self.root.config(bg="skyblue")
        self.image = ""
        self.image_path = ""
        self.hm = ""
        self.model = model

        window_width = 1500
        window_height = 850
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

        file_upload = ttk.Button(self.left_frame, text="Upload 2D Fundus Image",
                                 command=lambda: self.upload_file())
        file_upload.pack(fill="both", padx=20, pady=5)

        task_manager = tk.Frame(self.left_frame, width=400, height=300, bg="lightgrey")
        task_manager.pack(fill="both", padx=5, pady=5)

        run_gpp = ttk.Button(task_manager, text="Produce Heatmap of Anomalies",
                             command=lambda: self.run_gpp())
        run_gpp.pack(fill="both", padx=20, pady=5)

        run_pred = ttk.Button(task_manager, text="Predict DR severity level")
        run_pred.pack(fill="both", padx=20, pady=5)

        self.root.resizable(False, False)

    def run(self):
        self.root.mainloop()

    def upload_file(self):
        file_path = askopenfile(mode='r', filetypes=[('Image Files', '*jpeg')])
        if file_path is not None:
            pass
        self.image_path = file_path.name
        self.image = ImageTk.PhotoImage(Image.open(self.image_path))

        output = tk.Label(self.left_frame, text=file_path, image=self.image)
        output.pack(fill="both", padx=5, pady=5)

    def run_gpp(self):
        if self.image != "":
            heatmap = gradCAMplusplus(
                self.image_path, self.model.model, self.model.last_conv)
            overlayed = overlap_heatmap(self.image_path, heatmap, 0.3)

            self.image_hm = ImageTk.PhotoImage(overlayed)
            output_text = tk.Label(self.right_frame, text="Areas of caution")
            output_text.pack(fill="both", padx=5, pady=5)
            gpp_output = tk.Label(
                self.right_frame, text="Areas of caution", image=self.image_hm)
            gpp_output.pack(fill="both", padx=5, pady=5)
        else:
            error = tk.Label(self.right_frame,
                             text="Please select an image before trying anything")
            error.pack(fill="both", padx=5, pady=5)
