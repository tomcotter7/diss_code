import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from utils.explainbility_utils import gradCAMplusplus, overlap_heatmap


class IncResNetV2():

    def __init__(self, weights, training):
        self.weights = weights
        self.base_lr = 0.001
        self.model = self.__build_model(training)
        self.last_conv = list(filter(lambda x: isinstance(
            x, tf.keras.layers.Conv2D), self.model.layers))[-1].name
        if not training:
            self.full_model.load_weights(weights)
        self.cam = None
        self.history = None
        self.history_fine = None

    def update_weights(self, weights):
        self.weights = weights
        self.full_model.load_weights(weights)

    def train(self, train_ds, val_ds, callbacks, epochs, ft_epochs):
        self.history = self.model.fit(train_ds, epochs=epochs,
                                      callbacks=callbacks, validation_data=val_ds)
        self.__update_frozen_layers()
        new_epochs = epochs + ft_epochs
        self.history_fine = self.model.fit(
            train_ds, epochs=new_epochs, initial_epoch=self.history.epoch[-1],
            callbacks=callbacks, validation_data=val_ds)

    def __build_model(self, training):
        inputs = tf.keras.Input(shape=(512, 512, 3))
        base_model = InceptionResNetV2(
            include_top='False', weights='None',
            input_shape=(512, 512, 3), input_tensor=inputs)
        if training:
            base_model.trainable = False

        target_conv_layer = list(filter(lambda x: isinstance(
                               x, tf.keras.layers.Conv2D), base_model.layers))[-1].name

        conv_layer = base_model.get_layer(target_conv_layer)
        x = GlobalAveragePooling2D()(conv_layer.output)
        x = Dropout(0.3)(x)
        predictions = Dense(3, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=predictions)

        model.compile(
            optimizer=tf.keras.optimizers.Nadam(
                learning_rate=self.base_lr,
                beta_1=0.95),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def __update_frozen_layers(self):

        self.model.trainable = True

        for layer in self.model.layers[:300]:
            layer.trainable = False

        self.model.compile(optimizer=tf.keras.optimizers.Nadam(
                                            learning_rate=self.base_lr/10),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def run_gradcam_pp(self, image_path):
        self.heatmap = gradCAMplusplus(image_path, self.model, self.last_conv)
        self.overlayed_image = overlap_heatmap(image_path, self.heatmap, 0.3)
