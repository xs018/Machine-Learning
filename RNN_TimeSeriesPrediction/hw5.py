# from PIL import Image
# im = Image.open('DS-1_36W.01087.tif')
# im.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import math
from sklearn.metrics import mean_squared_error
from tensorflow.python.util.nest import flatten

np.random.seed(1234)
def create_RNN(hidden_units, time_steps):
    model = keras.Sequential()
    model.add(layers.LSTM(hidden_units, activation='relu'))
    model.add(layers.Dense(time_steps))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
 
# Parameter split_percent defines the ratio of training examples
# def get_train_test(url, split_percent=(0.7, 0.1, 0.2)):
#     df = pd.read_csv(url, usecols=[1], engine='python', sep = "\t")
#     data = np.array(df.values.astype('float32'))
#     n = len(data)
#     # Point for splitting data into train and test
#     train_data = data[0:int(n*split_percent[0])]
#     val_data = data[int(n*split_percent[0]): int(n*(split_percent[0] +  split_percent[1]))]
#     test_data = data[int(n*(split_percent[0] +  split_percent[1])):]
#     return train_data, val_data, test_data

def sample_overlapping_seq(dat, time_steps, stride=1):
    sq_T = dat.shape[0]
    print(sq_T)
    num_seq = (sq_T - time_steps) // stride + 1
    indices = (np.arange(time_steps) + np.arange(0, num_seq * stride, stride).reshape(-1,1)).reshape(-1, )
    print(indices)
    sliding_windows = dat[indices].reshape(-1, time_steps)
    print(sliding_windows)
    return sliding_windows[..., np.newaxis]

# Prepare the input X and target Y
def get_XY(data, time_steps, stride=1):
    # Indices of target array
    num_samples = len(data)
    X = sample_overlapping_seq(data[:num_samples-time_steps], time_steps, stride) 
    Y = sample_overlapping_seq(data[time_steps:], time_steps, stride) 
    return X, Y

time_steps = 100

## data normalization zscore
scaler = MinMaxScaler((0, 1))

# train_data = scaler.fit_transform(train_data)
# val_data = scaler.transform(val_data)
# test_data = scaler.transform(test_data)

# trainX, trainY = get_XY(train_data, time_steps)
# valX, valY = get_XY(val_data, time_steps)
# testX, testY = get_XY(test_data, time_steps, stride=time_steps)

df = pd.read_csv("DS-1_36W_vapor_fraction.txt", usecols=[1], engine='python', sep = "\t")
data = np.array(df.values.astype('float32'))
X, y = get_XY(data, time_steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

checkpoint_filepath ='res/hw5/best_model.hdf5'

model = create_RNN(hidden_units=128, time_steps=time_steps) 
monitor = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto', restore_best_weights=True)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
filepath=checkpoint_filepath,
save_weights_only=True,
monitor='val_loss',
mode='min',
save_best_only=True)
history = model.fit(X_train, y_train, batch_size=32, validation_split=0.2, shuffle=True, callbacks=[monitor,model_checkpoint_callback], verbose=2, epochs=100)
print(model.summary())
# make predictions
# train_predict = model.predict(trainX)

# print(X_test.shape,y_test.shape)
# model.load_weights(checkpoint_filepath)

y_pred = model.predict(X_test)
# train_predict = model.predict(X_train)

# print(y_pred.shape)
test_mse = ((y_test.flatten()-y_pred.flatten()) ** 2).mean()
print(f"test mean square error: {test_mse:.4f}")
# def print_error(y_train, y_test, train_predict, test_predict):    
#     # Error of predictions
#     train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
#     test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
#     # Print RMSE
#     print('Train RMSE: %.3f RMSE' % (train_rmse))
#     print('Test RMSE: %.3f RMSE' % (test_rmse))    

# # make predictions
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)
# print(y_train.shape, y_test.shape, train_predict.shape, test_predict.shape)
# Mean square error
# print_error(y_train, y_test, train_predict, test_predict)

plot_cases = 50
with open("res/hw5/pred.npy", "wb") as f:
    np.save(f, X_test)
    np.save(f, y_pred)
    np.save(f, y_test)

plt.figure()
plt.plot(y_pred[:plot_cases].reshape(-1, 1), label="Predicted Signal")
plt.plot(y_test[:plot_cases].reshape(-1, 1), label="True Signal")
plt.legend()
plt.savefig("res/hw5/prob4.jpg")
plt.close()