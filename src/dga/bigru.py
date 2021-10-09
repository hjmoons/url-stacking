# Load Libraries
import os

from keras import regularizers
from keras.layers import Input, Bidirectional
from keras.models import Model
from keras.layers import Embedding, GRU
from keras.layers.core import Dense, Dropout, Flatten


def define_model(max_len=73, emb_dim=32, max_vocab_len=39, W_reg=regularizers.l2(1e-4)):
    """BiGRU model with the Keras Sequential model"""
    input = Input(shape=(max_len,), dtype='int32', name='bigru_input')
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg)(input)
    emb = Dropout(0.5)(emb)
    bigru = Bidirectional(GRU(128))(emb)
    bigru = Dropout(0.5)(bigru)
    output = Dense(21, activation='softmax')(bigru)
    model = Model(input=[input], output=[output])
    return model


# 모델을 저장하지 않는 경우 사용
def train_model(x_train, y_train, epochs, batch_size):
    # Define Deep Learning Model
    model = define_model()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)
    return model


def save_model(x_train, y_train, x_test, y_test, epochs=5, batch_size=64, export_path='./output/gru'):
    model = train_model(x_train, y_train, epochs=epochs, batch_size=batch_size)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    model_json = model.to_json()
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    with open(export_path + "/gru.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(export_path + "/gru.h5")
    return acc