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

from model_preprocessor import Preprocessor

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

def load_models(fileModelJSON, fileWeights):
    with open(fileModelJSON, 'r') as f:
        model_json = f.read()
        model = model_from_json(model_json)
        f.close()
    model.load_weights(fileWeights)
    return model

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

models_dir = "./saved_model/"
CNNModel = load_models(models_dir + "cnn.json", models_dir + "cnn.h5")
LSTMModel = load_models(models_dir + "lstm.json", models_dir + "lstm.h5")
GRUModel = load_models(models_dir + "gru.json", models_dir + "gru.h5")

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

CNNModel.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
LSTMModel.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
GRUModel.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

members = list()
members.append(CNNModel)
members.append(LSTMModel)
members.append(GRUModel)

# define ensemble model
stacked_model = define_stacked_model(members)

print("################################")
print("Complete Define Stacked Model!!")

# fit stacked model on test dataset
print("#################################")
print("Start training!!!!")

fit_stacked_model(stacked_model, x_train, y_train)
print("Complete Train Stacked Model!!")

# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, x_test)
#yhat = argmax(yhat, axis=1)
print(yhat)
# Save test result
from sklearn.metrics import classification_report
print(classification_report(y_test, yhat.round(), target_names=['benign', 'malicious']))

for model in members:
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Model Accuracy: %.3f' % acc)

X = [x_test for _ in range(len(stacked_model.input))]
stacked_model.evaluate(X, y_test)

print(stacked_model.input)
print(stacked_model.output)
print(stacked_model.summary())

from keras import backend as K

sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

builder = tf.saved_model.builder.SavedModelBuilder("./output/model")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()

print(stacked_model.input)