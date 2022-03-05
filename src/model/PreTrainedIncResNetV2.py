import tensorflow as tf
import numpy as np
from utils.model_utils import build_model, preprocess_data_for_grad_cam
from utils.explainbility_utils import gradCAMplusplus
from utils.training_utils import data_augmentation


class PreTrainedIncResNetV2():

    def __init__(self, weights):
        self.weights = weights
        self.model = build_model(False)
        self.last_conv = list(filter(lambda x: isinstance(
            x, tf.keras.layers.Conv2D), self.model.layers))[-1].name
        self.full_model.load_weights(weights)

    def update_weights(self, weights):
        self.weights = weights
        self.full_model.load_weights(weights)

    def run_gradcam_pp(self, image_path):
        img = np.asarray(preprocess_data_for_grad_cam(image_path))
        img = data_augmentation(img)
        cam = gradCAMplusplus(img, self.model, self.last_conv)
        # still need to visualize the heatmap output
        return cam

    def visualize_output(self, gradcam):
        pass
