from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras as K
from model.model_utils import preprocess_data_for_grad_cam
from skimage.transform import resize
import os
import cv2 as cv
import numpy as np


# a main function to just get things tested - won't be used in the final repo


def dirty_main(my_weights, image_path):

    data_augmentation = K.Sequential([
        K.layers.RandomFlip('horizontal'),
        K.layers.RandomRotation(0.3),
        K.layers.Rescaling(1. / 127.5, offset=-1)
    ])

    inps = K.Input(shape=(512, 512, 3))

    base_model = InceptionResNetV2(
        include_top=False, weights=None, input_tensor=inps, input_shape=(512, 512, 3))

    target_conv_layer = list(filter(lambda x: isinstance(
                           x, tf.keras.layers.Conv2D), base_model.layers))[-1].name

    conv_layer = base_model.get_layer(target_conv_layer)
    x = GlobalAveragePooling2D()(conv_layer.output)
    x = Dropout(0.3)(x)
    predictions = Dense(3, activation="softmax")(x)

    model = Model(inputs=inps, outputs=predictions)

    #print(model.summary())

    model.load_weights(my_weights).expect_partial()

    target_conv_layer = list(filter(lambda x: isinstance(
                           x, tf.keras.layers.Conv2D), base_model.layers))[-1].name

    img = np.asarray(preprocess_data_for_grad_cam(image_path))
    img = data_augmentation(img)

    grad_cam_plus(img, model, target_conv_layer)


def grad_cam_plus(img, model, layer_name):
    img_tensor = np.expand_dims(img, axis=0)
    conv_layer = model.get_layer(layer_name)
    gradModel = tf.keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_output, preds = gradModel(img_tensor)
                print(preds[0])
                class_id = np.argmax(preds[0])
                output = preds[:, class_id]
                conv_first_grad = tape3.gradient(output, conv_output)
            conv_second_grad = tape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = tape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))
    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0] * 2.0 + global_sum * conv_third_grad[0]
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom

    relu_conv = np.maximum(conv_first_grad[0], 0.0)

    weights = np.sum(alphas*relu_conv, axis=(0, 1))
    forward_activation_maps = np.sum(weights*conv_output[0], axis=2)
    cam = np.maximum(forward_activation_maps, 0.0)  # passing through RELU
    cam = cam / np.max(cam)  # scale 0 to 1.0
    cam = resize(cam, (512, 512))
    print(cam)


dirty_main("weights/dbest_save_at_34.ckpt", "images/936_right.jpeg")
