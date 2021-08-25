# Load Libraries
import os
import warnings
import json
from pathlib import Path

import tensorflow as tf
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.models import model_from_json

from preprocessor import Preprocessor

warnings.filterwarnings("ignore")

class UrlStacking:
    def __init__(self):
        pass
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