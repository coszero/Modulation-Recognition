#!/usr/bin/python
# coding = utf-8

import numpy as np
from keras import metrics, regularizers, backend
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # load train data
    Nfeature = 5
    NClass = 4
    NTrain = 50000
    train_data = np.loadtxt('./dataset/train.csv', delimiter = ',', dtype = float)
    ydata = train_data[:,-1]
    xdata = train_data[:,2:Nfeature]  #only use 3 features-4 cum
    ## Data preprocessing
    index = np.arange(ydata.shape[0]) #shuffle the train dataset
    np.random.shuffle(index)
    xdata = xdata[index,:]
    ydata = ydata[index]
    ydata = np_utils.to_categorical(ydata, NClass) #dummy encoding
    xTrain = xdata[0:int(xdata.shape[0]*9/10),:]
    yTrain = ydata[0:int(xdata.shape[0]*9/10),:]  #seperate dataset
    xVal = xdata[int(xdata.shape[0]*9/10):-1,:]
    yVal = ydata[int(xdata.shape[0]*9/10):-1,:]

    ## Train model
    _in_ = Input(shape=(xTrain.shape[1],))
    ot = Dense(8, activation='relu')(_in_)
    ot = Dense(16, activation='relu')(ot)
    ot = Dense(32, activation='relu')(ot)
    ot = Dense(16, activation='relu')(ot)
    _out_ = Dense(NClass, activation='softmax')(ot)
    model = Model(_in_, _out_)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = ModelCheckpoint(filepath='./result/weights', monitor='val_loss', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    model.fit(xTrain, yTrain,
              epochs=200,
              batch_size=250,
              validation_data=(xVal, yVal),
              shuffle=False,
              verbose=2,
              callbacks=[checkpoint, early_stopping])
    print("evaluate the model - train_set:")
    model.summary()
    
    # test model
    LOW = -10
    HIGH = 20
    GAP = 2
    NTest = 1000
    test_data = np.loadtxt('./dataset/test.csv', delimiter = ',', dtype = float)
    #xTest = test_data[:,0:Nfeature]
    xTest = test_data[:,2:Nfeature]
    
    yTest = test_data[:,-2]
    yTest.astype(int)
    yPredict = model.predict(xTest)
    predict_curve = np.zeros((6, int((HIGH-LOW)/GAP + 1))) # snr + Pc 
    for i in range(predict_curve.shape[1]): # snr - [-10 -8 ... 18 20]
        predict_curve[0, i] = LOW + GAP * i
    for i in range(test_data.shape[0]): # Pc - count num [241 431 ... 3900 4000 4000]
        snr_loc = int((test_data[i, -1] - LOW)/GAP)
        if yTest[i] == np.argmax(yPredict[i,:]):
            predict_curve[1, snr_loc] = predict_curve[1, snr_loc] + 1
            predict_curve[int(2+yTest[i]), snr_loc] = predict_curve[int(2+yTest[i]), snr_loc] + 1
    for i in range(predict_curve.shape[1]):  # Pc - cal pc(cnt_num/sum_num 3900/4000)
        predict_curve[1, i] = predict_curve[1, i]/(NTest*NClass)
        for j in range(NClass):
            predict_curve[2+j, i] = predict_curve[2+j, i]/(NTest)

        
    np.savetxt('./result/CUM_NN_F3_L100_50000.txt', predict_curve, delimiter=',', fmt='%.6f')
    
