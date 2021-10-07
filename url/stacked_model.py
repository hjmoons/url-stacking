# Load Libraries
import tensorflow as tf
from keras import backend as K

import preprocessor

import warnings

from keras.models import Model
from keras.layers.core import Dense
from keras.layers import concatenate

import cnn, gru, lstm


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

    merge = concatenate(ensemble_outputs, name='final_input')
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(1, activation='sigmoid', name='final_output')(hidden)

    model = Model(inputs=ensemble_visible, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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


warnings.filterwarnings("ignore")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

epochs = 1
batch_size = 64

# Load data
x_train, x_test, y_train, y_test = preprocessor.load_data_binary(10000)

models = list()

models.append(cnn.define_model())
models.append(cnn.define_model())
models.append(cnn.define_model())
models.append(cnn.define_model())
models.append(lstm.define_model())
models.append(lstm.define_model())
models.append(gru.define_model())
models.append(gru.define_model())

acc_list = list()

print()
print("Start base model training!!!!")
print()

# train and evaluate base models
for model in models:
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)
    #model.predict(x_test, verbose=0)

    _, acc = model.evaluate(x_test, y_test, verbose=0)
    acc_list.append(acc)

print()
print("Complete base model training!!!!")
print()
print("Start stacking model training!!!!")
print()

# train and evaluate stacking ensemble model
stacked_model = define_stacked_model(models)
fit_stacked_model(stacked_model, x_train, y_train)
#stacked_pred = predict_stacked_model(stacked_model, x_test)

X = [x_test for _ in range(len(stacked_model.input))]
_, acc = stacked_model.evaluate(X, y_test, verbose=0)
acc_list.append(acc)

print()
print("Complete stacking model training!!!!")
print()

print('Model Accuracy')
print(acc_list)

'''
from keras import backend as K

sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

builder = tf.saved_model.builder.SavedModelBuilder("./output/model")
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
builder.save()

print(stacked_model.input)
'''