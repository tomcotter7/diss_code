import tensorflow as tf
import os


def define_datasets(batch_size, train_dir):
    image_size = (512, 512)

    os.listdir(train_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
          train_dir,
          validation_split=0.2,
          subset="training",
          label_mode="categorical",
          seed=1337,
          color_mode="rgb",
          image_size=image_size,
          batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          train_dir,
          validation_split=0.2,
          subset="validation",
          label_mode="categorical",
          seed=1337,
          color_mode="rgb",
          image_size=image_size,
          batch_size=batch_size,
     )

    data_augmentation = tf.keras.Sequential([
         tf.keras.layers.RandomFlip("horizontal_and_vertical"),
         tf.keras.layers.RandomRotation(0.2),
         tf.keras.layers.RandomContrast(0.15)
         ])

    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 5)
    val_ds = val_ds.skip(val_batches // 5)

    preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

    train_ds = train_ds.map(
      lambda x, y: (data_augmentation(x, training=True), y)
    )
    train_ds = train_ds.map(
      lambda x, y: (preprocess_input(x), y)
    )

    val_ds = val_ds.map(
      lambda x, y: (preprocess_input(x), y)
    )

    test_ds = test_ds.map(
      lambda x, y: (preprocess_input(x), y)
    )

    return train_ds, val_ds, test_ds
