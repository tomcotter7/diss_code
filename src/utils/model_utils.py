import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
import cv2 as cv


def build_model(training):

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

    base_learning_rate = 0.001
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(
            learning_rate=base_learning_rate,
            beta_1=0.95),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return base_model, model


def preprocess_data_for_grad_cam(img_path):
    img = cv.imread(img_path)
    img = crop_image_otsu(img)
    resized_img = cv.resize(img, (512, 512), interpolation=cv.INTER_LINEAR)
    cv.imwrite(img_path, resized_img)
    return img


def crop_image_otsu(img):

    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(grayscale, 0, 255, cv.THRESH_OTSU)
    bbox = cv.boundingRect(thresholded)
    x, y, w, h = bbox
    foreground = img[y:y+h, x:x+w]
    return foreground
