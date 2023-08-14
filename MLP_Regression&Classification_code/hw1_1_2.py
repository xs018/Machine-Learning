import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras import losses
import statistics
import time
import tensorflow as tf

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
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    start=time.time()
    model = create_mlp(1, True)
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                            patience=5, verbose=1, mode='auto', restore_best_weights=True)
    
    best_loss = np.Infinity
    best_losses_train = None
    best_losses_validation = None
    kf = KFold(100, shuffle=True, random_state=42)
    fold=0
    for train, test in kf.split(x):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        fold+=1
        print(f"Fold #{fold}")
        # print(history.history)
        history = model.fit(x_train,y_train, validation_data=(x_test,y_test), callbacks=[monitor], verbose=2, epochs=100) 
        if statistics.mean(history.history['val_loss']) < best_loss: 
            best_loss =  statistics.mean(history.history['val_loss'])
            best_losses_train = history.history['loss']
            best_losses_validation = history.history['val_loss']
    end=time.time()
    print(f"Eclapse time: {end-start}s")
    plt.figure()
    plt.plot(best_losses_train, label='training_loss')
    plt.plot(best_losses_validation, label='val_loss')
    plt.savefig('res/hw1_1_2mse.png')
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
