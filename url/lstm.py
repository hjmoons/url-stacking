# Load Libraries
from keras import regularizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Embedding, LSTM
from keras.layers.core import Dense, Dropout, Flatten


class UrlLSTM:
    def __init__(self):
        pass

    @staticmethod
    def lstm(max_len=80, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):
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

        return model
