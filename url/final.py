# Load Libraries
import os
from pathlib import Path
import json

import tensorflow as tf

from keras.models import Model
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.models import model_from_json

from preprocessor import Preprocessor

import warnings

from url.cnn import UrlCNN
from url.gru import UrlGRU
from url.lstm import UrlLSTM
from url.stacking import UrlStacking

warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))


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

    merge = concatenate(ensemble_outputs, name='final_input')
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(1, activation='sigmoid', name='final_output')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


    # fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data

    # fit model
    model.fit(X, inputy, epochs=10, batch_size=64)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


# Load data
x_train, x_test, y_train, y_test = Preprocessor.load_data_binary(10000)

print("##################")
print("Data preprocessing")

models = list()

models.append(UrlCNN.cnn())
models.append(UrlLSTM.lstm())
models.append(UrlGRU.gru())

for model in models:
    epochs = 5
    batch_size = 64
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)

# define ensemble model
#stacked_model = define_stacked_model(models)
stacked_model = UrlStacking.define_stacked_model(models)

print("################################")
print("Complete Define Stacked Model!!")

# fit stacked model on test dataset
print("#################################")
print("Start training!!!!")

fit_stacked_model(stacked_model, x_train, y_train)
print("Complete Train Stacked Model!!")


# make predictions and evaluate
cnn_pred = model.predict(x_test, verbose=0)
lstm_pred = model.predict(x_test, verbose=0)
gru_pred = model.predict(x_test, verbose=0)
stacked_pred = predict_stacked_model(stacked_model, x_test)

print("######## result")
# Save test result
'''
from sklearn.metrics import classification_report

print(classification_report(y_test, cnn_pred.round(), target_names=['benign', 'malicious']))
print(classification_report(y_test, lstm_pred.round(), target_names=['benign', 'malicious']))
print(classification_report(y_test, gru_pred.round(), target_names=['benign', 'malicious']))
print(classification_report(y_test, stacked_pred.round(), target_names=['benign', 'malicious']))
'''

for model in models:
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print(model.input)
    print('Model Accuracy: %.3f' % acc)

X = [x_test for _ in range(len(stacked_model.input))]
_, acc = stacked_model.evaluate(X, y_test, verbose=0)
print('Model Accuracy: %.3f' % acc)

from keras import backend as K

sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

builder = tf.saved_model.builder.SavedModelBuilder("./output/model")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()

print(stacked_model.input)