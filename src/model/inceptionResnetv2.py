from incresv2_utils import build_model, preprocess_data
import cv2 as cv
import numpy as np


class IncResNetV2():
    # define an __init__ to load in the weights
    # weights will be stored in ../models/final_weights.ckpt
    # model should be fully defined to test against 2D fundus images
    def __init__(self, weights):
        self.model = build_model(weights)

    def update_weights(self, new_weights):
        self.model = build_model(new_weights)

    # function to run an image through the model

    def run_image(self, image):
        img = cv.imread(image)
        img = preprocess_data(img)
        preds = self.model.predict(img)
        i = np.argmax(preds[0])
        # here need to run GradCAM++ on i, self.model, target conv layer
        return i
