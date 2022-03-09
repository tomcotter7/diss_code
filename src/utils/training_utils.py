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


def define_datasets(batch_size, train_dir):
    image_size = (512, 512)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
          train_dir,
          validation_split=0.2,
          subset="training",
          seed=1337,
          color_mode="rgb",
          image_size=image_size,
          batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          train_dir,
          validation_split=0.2,
          subset="validation",
          seed=1337,
          color_mode="rgb",
          image_size=image_size,
          batch_size=batch_size,
     )

    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 5)
    val_ds = val_ds.skip(val_batches // 5)

    # these three lines make the datasets smaller - using for hyper parameter testing
    train_ds = train_ds.take(tf.data.experimental.cardinality(train_ds) // 2)
    val_ds = val_ds.take(tf.data.experimental.cardinality(val_ds) // 2)
    test_ds = test_ds.take(tf.data.experimental.cardinality(test_ds) // 2)

    return train_ds, val_ds, test_ds
