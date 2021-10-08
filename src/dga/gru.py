# Load Libraries
import os

from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Embedding, GRU
from keras.layers.core import Dense, Dropout, Flatten


def define_model(max_len=73, emb_dim=32, max_vocab_len=40, W_reg=regularizers.l2(1e-4)):
    """GRU model with the Keras Sequential model"""
    input = Input(shape=(max_len,), dtype='int32', name='gru_input')
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg)(input)
    emb = Dropout(0.5)(emb)
    gru = GRU(128)(emb)
    gru = Dropout(0.5)(gru)
    output = Dense(21, activation='softmax', name='gru_output')(gru)
    model = Model(input=[input], output=[output])
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# 모델을 저장하지 않는 경우 사용
def train_model(x_train, y_train, epochs, batch_size):
    # Define Deep Learning Model
    model = define_model()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)
    return model


def save_model(x_train, y_train, epochs=5, batch_size=64, export_path='./output/gru'):
    model = train_model(x_train, y_train, epochs=epochs, batch_size=batch_size)
    model_json = model.to_json()
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    with open(export_path + "/gru.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(export_path + "/gru.h5")