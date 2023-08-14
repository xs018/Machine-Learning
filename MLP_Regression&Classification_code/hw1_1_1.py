import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import time 

tf.random.set_seed(1234)
def create_mlp(dim, regress=False):
    model = Sequential()
    model.add(Dense(256, input_dim=dim, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    if regress:
        model.add(Dense(1, activation="linear"))
    return model

def main():
    file_path = '/ocean/projects/mch210006p/shared/HW1/Regression/boiling-32_temp_heat_flux.txt'
    data = pd.read_csv(file_path, delimiter = "\t", names=['Temperature(C)', 'Heat flux(W/cm2)'], header=0)
    x = zscore(data['Temperature(C)'])
    y = data['Heat flux(W/cm2)'].values
    start=time.time()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = create_mlp(1, True)
    print(model.summary())
    # rmse = tf.keras.metrics.RootMeanSquaredError()
    model.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                            patience=5, verbose=1, mode='auto', restore_best_weights=True)
    history = model.fit(x_train,y_train, validation_data=(x_test,y_test), callbacks=[monitor], verbose=2, epochs=100)     
    end=time.time()
    print(f"Eclapse time: {end-start}s")

    # print(history.history)
    plt.figure()
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.savefig('res/hw1_1_1mse.png')
    # plt.figure()
    # plt.plot(history.history['rmse'], label='training accuracy')
    # plt.plot(history.history['val_rmse'], label='val_accuracy')
    # plt.savefig('res/hw1_1_rmse.png')
    
    # Predict
    pred = model.predict(x_test)

    # Measure MSE error. 
    score_mse = metrics.mean_squared_error(pred, y_test)

    # Measure RMSE error.  RMSE is common for regression.
    score_rmse = np.sqrt(score_mse)

    print(f"Mean Square Error: {score_mse}")
    print(f"Rooted Mean Square Error: {score_rmse}")

    

if __name__ == '__main__':
    main()
