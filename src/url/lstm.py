# Load Libraries
import os

from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Embedding, LSTM
from keras.layers.core import Dense, Dropout, Flatten


def define_model(max_len=80, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):
        """LSTM with Attention model with the Keras Sequential model"""

        input = Input(shape=(max_len,), dtype='int32', name='lstm_input')

        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg)(input)
        emb = Dropout(0.2)(emb)

        lstm = LSTM(units=128, return_sequences=True)(emb)
        lstm = Dropout(0.5)(lstm)
        lstm = Flatten()(lstm)

        h1 = Dense(8576, activation='relu')(lstm)
        h1 = Dropout(0.5)(h1)

        output = Dense(1, activation='sigmoid', name='lstm_output')(h1)

        model = Model(input=[input], output=[output])

        model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

        return model


# 모델을 저장하지 않는 경우 사용
def train_model(x_train, y_train, epochs, batch_size):
        # Define Deep Learning Model
        model = define_model()
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)
        return model


def save_model(x_train, y_train, epochs=5, batch_size=64, export_path='./output/lstm'):
        model = train_model(x_train, y_train, epochs=epochs, batch_size=batch_size)
        model_json = model.to_json()
        if not os.path.exists(export_path):
                os.makedirs(export_path)
        with open(export_path + "/lstm.json", "w") as json_file:
                json_file.write(model_json)
        model.save_weights(export_path + "/lstm.h5")