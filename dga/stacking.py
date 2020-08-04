# Load Libraries
import pandas as pd
import numpy as np
import re, os
from string import printable
from sklearn import model_selection

import tensorflow as tf
from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
    BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import KFold
from pathlib import Path
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils import plot_model
from tensorflow.python.platform import gfile
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
from keras.utils import to_categorical
from numpy import dstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from numpy import argmax
from keras.models import model_from_json

import json

import warnings
warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

# General save model to disk function
def save_model(fileModelJSON, fileWeights):
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = stacked_model.to_json()
    with open(fileModelJSON, 'w') as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    stacked_model.save_weights(fileWeights)

# General load model from disk function
def load_models(fileModelJSON, fileWeights):
    with open(fileModelJSON, 'r') as f:
        model_json = json.load(f)
        model = model_from_json(model_json)

    model.load_weights(fileWeights)
    return model



with tf.device("/GPU:1"):
    def define_stacked_model(members):
        # update all layers in all models to not be trainable
        for i in range(len(members)):
            model = members[i]
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unique layer name' issue
                layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
        # define multi-headed input
        ensemble_visible = [model.input for model in members]
        # concatenate merge output from each model
        ensemble_outputs = [model.output for model in members]

        merge = concatenate(ensemble_outputs)
        hidden = Dense(10, activation='relu')(merge)
        output = Dense(20, activation='softmax')(hidden)
        model = Model(inputs=ensemble_visible, outputs=output)
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

        # fit a stacked model
    def fit_stacked_model(model, inputX, inputy):
        # prepare input data
        X = [inputX for _ in range(len(model.input))]
        # encode output data
        inputy_enc = to_categorical(inputy)

        # fit model
        model.fit(X, inputy, epochs=10, verbose=0, batch_size=64)

    # make a prediction with a stacked model
    def predict_stacked_model(model, inputX):
        # prepare input data
        X = [inputX for _ in range(len(model.input))]
        # make prediction
        return model.predict(X, verbose=0)

with tf.device("/GPU:1"):

    # Load data
    DATA_HOME ='/home/jhnamgung/kcyber/data/'
    df = pd.read_csv(DATA_HOME + 'dga_1st_round_train.csv',encoding='ISO-8859-1', sep=',')

    url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.domain]
    max_len = 74

    X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    y = np.array(df.nclass)

    X_train, X_test, y_train0, y_test0 = model_selection.train_test_split(X, y, test_size=0.2, random_state=33)

    y_train = np_utils.to_categorical(y_train0, 20)
    y_test = np_utils.to_categorical(y_test0, 20)

    print("##################")
    print("Data preprocessing")


    models_dir = "../models/"
    CNNmodel = load_models(models_dir + "1DCNN.json", models_dir + "1DCNN.h5")
    LSTMmodel = load_models(models_dir + "LSTM.json", models_dir + "LSTM.h5")
    CNNLSTMmodel = load_models(models_dir + "1DCNNLSTM.json", models_dir + "1DCNNLSTM.h5")
    BILSTMmodel = load_models(models_dir + "BILSTM.json", models_dir + "BILSTM.h5")

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    CNNmodel.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    LSTMmodel.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    CNNLSTMmodel.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    BILSTMmodel.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    members = list()
    members.append(CNNmodel)
    members.append(LSTMmodel)
    members.append(CNNLSTMmodel)
    members.append(BILSTMmodel)

    #for model in members:
    #    testy_enc = to_categorical(y_test)
    #    _, acc = model.evaluate(x_test, y_test, verbose=0)
    #    print('Model Accuracy: %.3f' % acc)

    epochs = 4
    batch_size = 32

    # define ensemble model
    stacked_model = define_stacked_model(members)
    
    print("################################")
    print("Complete Define Stacked Model!!") 


    # fit stacked model on test dataset
    print("#################################")
    print("Start training!!!!")

    fit_stacked_model(stacked_model, X_test, y_test)
    print("Complete Train Stacked Model!!")

   # make predictions and evaluate
    yhat = predict_stacked_model(stacked_model, X_test)
    yhat = argmax(yhat, axis=1)
    
    dgalist = []
    for x in yhat.tolist():
        if x != 0:
            dgalist.append("yes")
        else:
            dgalist.append("no")

    # Save test result
#    x_input = df_Test['domain'].tolist()
#    archive = pd.DataFrame(columns=['domain'])
#    archive["domain"] = x_input
#    archive["dga"] = dgalist
#    archive["class"] = yhat.tolist()
#    archive.to_csv("./testResult.csv",mode='w')
#    print(yhat)

    acc = accuracy_score(y_test0, yhat)
    print('Stacked Test Accuracy: %.3f' % acc)
    model_name = "STACKING"
    save_model("../models/" + model_name + ".json", "../models/" + model_name + ".h5")

