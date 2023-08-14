import pandas as pd
import numpy as np
import os
import pickle
from glob import glob
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve

np.random.seed(0)

def fit_pca(X_train, X_test, n_components):
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train_pcs = pca.fit_transform(X_train_std)
    X_test_pcs = pca.transform(X_test_std)
 
    return X_train_pcs, X_test_pcs

def create_mlp(hiddens=(8, 4), n_class=2):
    model = Sequential()
    model.add(Flatten())

    for i in range(len(hiddens)):
        model.add(Dense(hiddens[i], activation='relu'))

    model.add(Dense(n_class))
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model

def train(model, X_train_pcs, y_train, checkpoint_filepath, epoch=100):
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, 
                            patience=5, verbose=1, mode='auto', restore_best_weights=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    history = model.fit(X_train_pcs, y_train, validation_split=0.2, shuffle=True, callbacks=[monitor,model_checkpoint_callback], verbose=2, epochs=epoch)
    return history.history

def test(model, X_test_pcs):
    prediction = model.predict(X_test_pcs) 
    score = tf.nn.softmax(prediction).numpy()
    pred = np.argmax(score, axis=1) 
    return pred, score

# n_components = 100

# image_size = (240, 240)
# dataset = pd.read_csv("dataset.csv")
# n_samples = len(dataset)
# # idx = np.random.choice(len(dataset), n_samples)
# # selected = dataset.iloc[idx]
# data = np.empty((n_samples, image_size[0]*image_size[1]))
# y = dataset.iloc[:, 1].values
# for i in range(n_samples):
#     path = dataset.iloc[i, 0]
#     img = np.float32(imread(path)) / 255.
#     image_resized = resize(img, image_size, anti_aliasing=True)
#     data[i] = image_resized.flatten()

# # with open('dataset.npy', 'wb') as f:
# #     np.save(f, data)
# #     np.save(f, y)

# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state = 0)
# print(X_train.shape, y_train.shape)

# X_train_pcs, X_test_pcs = fit_pca(X_train, X_test, n_components)
# print(X_train_pcs.shape, X_test_pcs.shape, y.shape)

# save_path = '/ocean/projects/mch210006p/shared'
# with open(f'{save_path}/train_pca{n_components}.npy', 'wb') as f:
#     np.save(f, X_train_pcs)
#     np.save(f, y_train)

# with open(f'{save_path}/test_{n_components}.npy', 'wb') as f:
#     np.save(f, X_test_pcs)
#     np.save(f, y_test)

base_path = '/ocean/projects/mch210006p/shared'
n_components = 100
with open(f'{base_path}/train_pca{n_components}.npy', 'rb') as f:
    X_train_pcs = np.load(f)
    y_train = np.load(f)

print(X_train_pcs.shape, y_train.shape)

model = create_mlp(hiddens=(64, 32, 16))
checkpoint_filepath ='res/hw4/best_model.hdf5'
history = train(model, X_train_pcs, y_train, checkpoint_filepath, epoch=50)

plt.figure()
plt.plot(history['loss'], label='training_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel("Epoch")
plt.xlabel("Cross-Entropy Loss")
plt.legend()
plt.savefig('res/hw4/pca_mlp_train_loss.png')

plt.figure()
plt.plot(history['accuracy'], label='training_accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.xlabel("Epoch")
plt.xlabel("Accuracy")
plt.legend()
plt.savefig('res/hw4/pca_mlp_train_acc.png')

# checkpoint_filepath ='res/hw4/best_model.hdf5'
# model=create_mlp()
# model.load_weights(checkpoint_filepath)

with open(f'{base_path}/test_{n_components}.npy', 'rb') as f:
    X_test_pcs = np.load(f)
    y_test = np.load(f)

pred, score = test(model, X_test_pcs)

cm = confusion_matrix(y_test, pred, labels=[0, 1])

tn, fp, fn, tp = cm.ravel() # where 1 is positive, 0 is negative
print(f"True Negative: {tn}, False Positive: {fp}, False Negative: {fn}, True Postive: {tp}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['post-CHF(0)', 'pre-CHF(1)'])
disp.plot()
plt.savefig('res/hw4/hw4confusion_matrix.png')

fpr, tpr, thresholds = roc_curve(y_test, pred)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.savefig('res/hw4/hw4ROC.png')

prec, recall, _  = precision_recall_curve(y_test, pred)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
plt.savefig('res/hw4/hw4confusion_prediction.png')

print(f"Area Under Curve: {auc(fpr, tpr)}")
print(f"Accuracy: {(tp+tn) / (tn + fp + fn+ tp)}")
print(f"Precision: {(tp) / ( fp +  tp)}")
print(f"Recall: {(tp) / ( fn +  tp)}")
print(f"F1 Score: {tp / (tp + (fn + fp)/2)}")