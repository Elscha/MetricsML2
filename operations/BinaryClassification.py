'''
    Highlevel API to run binary classification with Keras
    Created on 15.03.2019

    @author: el-sharkawy
'''
import numpy as np 
import tensorflow as tf      # Tensorflow
from tensorflow import keras # Simplified Tensorflow Framework
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

def binaryClassification(data, labels, hiddenLayers, lrate, nEpochs, kSplitt=10, rp=0.01, columns=None, plotName=None):
    if (columns is not None):
        data = data[:, columns]
    
    if (kSplitt > 0):
        randomSeed = 0
        if (randomSeed != 0):
            kfold = StratifiedKFold(n_splits=kSplitt, shuffle=True, random_state=randomSeed)
        else:
            kfold = StratifiedKFold(n_splits=kSplitt, shuffle=True)
    
    i = 0;
    cvscores = []
    # K-Fold analysis based on https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    for train, test in kfold.split(data, labels):
        i = i+1
        ### Define Neuronal Network
        cbks = [callbacks.TerminateOnNaN()]
        layers=[keras.layers.Dense(i, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(rp)) for i in hiddenLayers]
#         layers=[keras.layers.Dense(i, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(rp)) for i in hiddenLayers]
#         layers=keras.layers.Dense(i, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(rp))(layers)
        layers.append(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        model = keras.Sequential(layers)
        
        model.compile(optimizer = tf.train.AdamOptimizer(),
                      lr        = lrate, 
                      loss      = 'binary_crossentropy',
                      metrics   = ['accuracy'])
        
        ### Execute model
        history =  model.fit(data[train], labels[train], epochs=nEpochs, callbacks=cbks, verbose=0) #validation_data=[test_data,test_labels]) #--> Use this to grep & plot this per Epochs (last line)
        scores = model.evaluate(data[test], labels[test], verbose=0)
    
        if (np.isnan(history.history['loss']).any()):
            raise ValueError("Loss was not a number")
        
        # Needs to be refactored
        if (plotName is not None):
            plt.plot(history.history['acc'])     
            #plt.plot(history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.savefig("../data/" + plotName + str(i) + ".png")
        
        
        print("%s %s: %.2f%%" % (i, model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))