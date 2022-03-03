import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
import cv2 as cv


def build_model(weights):

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.Rescaling(1./127.5, offset=-1)
    ])

    base_model = InceptionResNetV2(
        include_top='False', weights='None', input_shape=(512, 512, 3))
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(512, 512, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu")(x)
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

    model.load_weights(weights)

    return model


def preprocess_data(image_path):
    img = crop_image_otsu(image_path)
    h, w, d = img.size
    aspect_ratio = w / h
    width = 512
    height = int(aspect_ratio * width)
    size = (width, height)
    resized_img = cv.resize(img, size, interpolation=cv.INTER_NEAREST)
    cv.imwrite(image_path, resized_img)


def crop_image_otsu(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(grayscale, 0, 255, cv.THRESH_OTSU)
    bbox = cv.boundingRect(thresholded)
    x, y, w, h = bbox
    foreground = img[y:y+h, x:x+w]
    return foreground
