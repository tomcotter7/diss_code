from model.model_utils import build_model, preprocess_data
import tensorflow as tf


class PreTrainedIncResNetV2():

    def __init__(self, weights):
        self.weights = weights
        self.base, self.full_model = build_model(False)
        self.last_conv = list(filter(lambda x: isinstance(
            x, tf.keras.layers.Conv2D), self.base.layers))[-1].name
        self.full_model.load_weights(weights)

    def update_weights(self, weights):
        self.weights = weights
        self.full_model.load_weights(weights)

    def run_image(self, image_path):
        preprocess_data(image_path)
        dataset = tf.data.Dataset.list_files(image_path)
        preds = self.full_model.predict(dataset)
        return preds
