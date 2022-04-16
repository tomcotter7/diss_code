import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import time


# hyper parameters need fixing for this
class IncResNetV2():

    def __init__(self, weights, training):
        self.weights = weights
        self.base_lr = 0.001
        self.model = self.__build_model(training)
        self.last_conv = list(filter(lambda x: isinstance(
            x, tf.keras.layers.Conv2D), self.model.layers))[-1].name
        if not training:
            load = self.model.load_weights(weights).expect_partial()
            load.assert_existing_objects_matched()
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

        learning_rate = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001, first_decay_steps=1500)
        METRICS = [
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]

        inputs = tf.keras.Input(shape=(512, 512, 3))
        if training:
            base_model = InceptionResNetV2(
                include_top=False, weights='imagenet',
                input_shape=(512, 512, 3), input_tensor=inputs)
        else:
            base_model = InceptionResNetV2(
                include_top=False, weights=None,
                input_shape=(512, 512, 3), input_tensor=inputs)
        if training:
            base_model.trainable = False

        target_conv_layer = list(filter(lambda x: isinstance(
                               x, tf.keras.layers.Conv2D), base_model.layers))[-1].name
        conv_layer = base_model.get_layer(target_conv_layer)
        x = GlobalAveragePooling2D()(conv_layer.output)
        x = tf.keras.layers.GaussianNoise(0.5)(x)
        x = Dense(512, activation="relu", kernel_regularizer="l1")(x)
        x = Dropout(0.3)(x)
        predictions = Dense(3, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=predictions)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[METRICS])

        return model

    def __update_frozen_layers(self):

        self.model.trainable = True

        for layer in self.model.layers[:300]:
            layer.trainable = False

        self.model.compile(optimizer=tf.keras.optimizers.Nadam(
                                            learning_rate=self.base_lr/10),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
