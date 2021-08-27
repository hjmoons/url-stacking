# Load Libraries
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from preprocessor import Preprocessor

class UrlLSTM:
    def __init__(self):
        self.model = self.layer()

    @staticmethod
    def layer(max_len=80, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):
        input = Input(shape=(max_len,), dtype='int32', name='lstm_input')

        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, embeddings_regularizer=W_reg)(input)
        emb = Dropout(0.2)(emb)

        lstm = LSTM(units=128, return_sequences=True)(emb)
        lstm = Dropout(0.5)(lstm)
        lstm = Flatten()(lstm)

        h1 = Dense(8576, activation='relu')(lstm)
        h1 = Dropout(0.5)(h1)

        output = Dense(1, activation='sigmoid', name='lstm_output')(h1)

        model = Model(input, output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def save(self):
        model_json = self.model.to_json()
        with open("./saved_model/cnn.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("./saved_model/cnn.h5")