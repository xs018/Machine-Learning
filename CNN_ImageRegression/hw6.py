import os
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

base_path = "/ocean/projects/mch210006p/shared/HW5"
save_path = "/ocean/projects/mch210006p/xs018"
image_size = (240, 240)

#num_samples = 4999

#labels = pd.read_csv(os.path.join(base_path, "DS-1_36W_vapor_fraction.txt"), sep = "\t", usecols=[1])
#labels = labels.values

#imgs = []
#for idx in range(1, num_samples+1):
#    img_dir = os.path.join(base_path, "DS-1_36W_images/DS-1_36W.{:05d}.TIFF".format(idx))
#    img = Image.open(img_dir)
#    img = np.float32(np.array(img)) / 255.
#    img = resize(img, image_size, anti_aliasing=True)
#    imgs.append(img[..., np.newaxis])

#imgs = np.array(imgs)
#with open(os.path.join(save_path, "train.npy"), "wb") as f:
#    np.save(f, imgs)
#    np.save(f, labels)

def create_model(img_size=(240, 240)):
    input_shape = (img_size[0], img_size[1], 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mean_squared_error',
        metrics=['mse'])

    return model

with open(os.path.join(save_path, "train.npy"), "rb") as f:
    imgs = np.load(f)
    labels = np.load(f)

mean = imgs.mean()
std = imgs.std()
data = (imgs - mean) / std

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

checkpoint_filepath = f'{save_path}/best_model.hdf5'
monitor = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                monitor='val_loss',
                                                                mode='min',
                                                                save_best_only=True)
model = create_model()

history = model.fit(X_train, y_train, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[monitor, model_checkpoint_callback], verbose=2, epochs=100)
print(model.summary())

plt.figure()
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="validation loss")
plt.xlabel('epoches')
plt.ylabel('loss')
plt.legend()
plt.savefig("res/hw6/loss.png")

y_pred = model.predict(X_test)
test_mse = ((y_test.flatten()-y_pred.flatten()) ** 2).mean()
print(f"test mean square error: {test_mse:.4f}")

