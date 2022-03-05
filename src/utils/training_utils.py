import tensorflow as tf


def data_augmentation(dataset):

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.Rescaling(1./127.5, offset=-1)
    ])

    return dataset.map(
        lambda x, y:  (data_augmentation(x, training=True), y)
    )
