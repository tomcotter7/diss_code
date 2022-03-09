
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model.IncResNetV2 import IncResNetV2
from utils.training_utils import data_augmentation, define_datasets
from data_preprocessing import run_pp

src_path = "/content/drive/MyDrive/year3/diss/train/"
dst_path = "/content/drive/MyDrive/year3/diss/512-train/"
src_labels = "/content/drive/MyDrive/year3/diss/trainLabels.csv"

checkpoint_path = "/content/drive/MyDrive/year3/diss/models/dbest_save_at_{epoch}.ckpt"

run_pp(src_path, dst_path, src_labels)


def callbacks(chck_path):

    callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      verbose=1,
      monitor='val_accuracy',
      mode='max',
      save_weights_only=True,
      save_best_only=True,
    )

    return callback


model = IncResNetV2(None, True)
train_ds, val_ds, test_ds = define_datasets(dst_path)
model.train(train_ds, val_ds, [callbacks()], 25, 10)

# here show plots for how accuary and loss improved through transfer learning and fine tuning

acc = model.history.history['accuracy']
val_acc = model.history.history['val_accuracy']

loss = model.history.history['loss']
val_loss = model.history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.5])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


acc = model.history_fine.history['accuracy']
val_acc = model.history_fine.history['val_accuracy']

loss = model.history_fine.history['loss']
val_loss = model.history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.5])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
