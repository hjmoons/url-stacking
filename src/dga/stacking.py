# Load Libraries
import os, warnings
from pathlib import Path

import tensorflow as tf
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers.core import Dense
from keras.layers import concatenate

import cnn, gru, lstm, preprocessor
import bigru, bilstm

warnings.filterwarnings("ignore")


def load_models(fileModelJSON, fileWeights):
    with open(fileModelJSON, 'r') as f:
        model_json = f.read()
        model = model_from_json(model_json)
        f.close()
    model.load_weights(fileWeights)
    return model


def define_stacked_model(models):
    # update all layers in all models to not be trainable
    for i in range(len(models)):
        model = models[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in models]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in models]

    merge = concatenate(ensemble_outputs)
    hidden = Dense(32, activation='relu')(merge)
    output = Dense(21, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, inputX, inputy, epochs=5, batch_size=64):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data

    # fit model
    model.fit(X, inputy, epochs=epochs, batch_size=batch_size)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)


def save_stacked_model(sess, export_path='./output/stacking'):
    init = tf.global_variables_initializer()
    sess.run(init)

    if Path(export_path).is_dir():
        os.remove(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
    builder.save()


def set_base_models(models_dir='./output'):
    models = list()

    CNNmodel = load_models(models_dir + "/cnn/cnn.json", models_dir + "/cnn/cnn.h5")
    LSTMmodel = load_models(models_dir + "/lstm/lstm.json", models_dir + "/lstm/lstm.h5")
    GRUmodel = load_models(models_dir + "/gru/gru.json", models_dir + "/gru/gru.h5")
    BiGRUmodel = load_models(models_dir + "/bigru/bigru.json", models_dir + "/bigru/bigru.h5")
    BiLSTMmodel = load_models(models_dir + "/bilstm/bilstm.json", models_dir + "/bilstm/bilstm.h5")

    CNNmodel.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    LSTMmodel.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    GRUmodel.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    BiGRUmodel.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    BiLSTMmodel.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    models.append(CNNmodel)
    models.append(LSTMmodel)
    models.append(GRUmodel)

    return models


def train_base_models(data_path, epochs=5):
    acc_list = list()

    # sampling data for training base models
    x_train, x_test, y_train, y_test = preprocessor.load_data(data_path)

    if not Path('./output').is_dir():   os.makedirs('./output')
    if Path('./output/cnn').is_dir():   os.remove('./output/cnn')
    if Path('./output/lstm').is_dir():  os.remove('./output/lstm')
    if Path('./output/gru').is_dir():   os.remove('./output/gru')
    if Path('./output/bilstm').is_dir():  os.remove('./output/bilstm')
    if Path('./output/bigru').is_dir():   os.remove('./output/bigru')

    acc_list.append(cnn.save_model(x_train, y_train, x_test, y_test, epochs=epochs))
    acc_list.append(lstm.save_model(x_train, y_train, x_test, y_test, epochs=epochs))
    acc_list.append(gru.save_model(x_train, y_train, x_test, y_test, epochs=epochs))
    acc_list.append(bilstm.save_model(x_train, y_train, x_test, y_test, epochs=epochs))
    acc_list.append(bigru.save_model(x_train, y_train, x_test, y_test, epochs=epochs))

    return acc_list


# parameter setting
epochs = 5
batch_size = 64
data_path = '../data/dga_label.csv'

# train and evaluate base models
acc_list = train_base_models(data_path, epochs)
models = set_base_models()

# Load data for training stacking model
x_train, x_test, y_train, y_test = preprocessor.load_data(data_path)

# train and evaluate stacking ensemble model
stacked_model = define_stacked_model(models)
fit_stacked_model(stacked_model, x_train, y_train, epochs=1)

X = [x_test for _ in range(len(stacked_model.input))]
_, acc = stacked_model.evaluate(X, y_test, verbose=0)
acc_list.append(acc)

# print accuracy of all models
print('Model Accuracy')
print(acc_list)

# save stacking model
save_stacked_model(K.get_session())


