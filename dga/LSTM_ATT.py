import warnings

import tensorflow as tf
from model_evaluator import Evaluator
from model_preprocessor import Preprocessor
from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention
import os
os.environ['TF_KERAS'] = '1'
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# GPU memory setting
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

warnings.filterwarnings('ignore')


def lstm_with_attention(max_len=74, emb_dim=32, max_vocab_len=40, W_reg=regularizers.l2(1e-4)):
    # """LSTM with Attention model with the Keras Sequential API"""

    model = Sequential()
    model.add(Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add((SeqSelfAttention(attention_activation='relu')))
    model.add(Flatten())
    model.add(Dense(9472, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(21, activation='softmax'))

    return model

x_train, x_test, y_train, y_test = Preprocessor.load_data()

model_name = "LSTM_ATT"
model = lstm_with_attention()
model.summary()

# Plot model(.png)
# tf.keras.utils.plot_model(model, to_file="./result/" + model_name + '.png', show_shapes=True)

epochs = 10
batch_size = 64
adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
checkpoint_filepath = './tmp/checkpoint/' +  model_name + '.hdf5'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',
                                                               save_best_only=True, mode='auto')

model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(),
                       Evaluator.precision, Evaluator.recall, Evaluator.fmeasure])

history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11, callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)   # Error in Windows environment

# Validation curves
Evaluator.plot_validation_curves(model_name, history)

y_pred = model.predict(x_test, batch_size=64)

# Experiment result
Evaluator.calculate_measure(model, x_test, y_test)
Evaluator.plot_confusion_matrix(model_name, y_test, y_pred)
Evaluator.plot_roc_curves(model_name, y_test, y_pred)