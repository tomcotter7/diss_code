from model.model_utils import build_model, preprocess_data
import tensorflow as tf
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

    def run_image(self, image_path):
        preprocess_data(image_path)
        dataset = tf.data.Dataset.list_files(image_path)
        preds = self.model.predict(dataset)
        # here need to run GradCAM++ on i, self.model, target conv layer
        return preds
