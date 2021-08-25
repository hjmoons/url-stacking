from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Flatten


class UrlGRU:
    def __init__(self):
        self.model = self.gru()

    @staticmethod
    def model(max_len=80, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):
        input = Input(shape=(max_len,), dtype='int32', name='gru_input')

        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                        embeddings_regularizer=W_reg)(input)
        emb = Dropout(0.2)(emb)

        gru = GRU(units=128, return_sequences=True)(emb)
        gru = Dropout(0.5)(gru)
        gru = Flatten()(gru)

        h1 = Dense(8576, activation='relu')(gru)
        h1 = Dropout(0.5)(h1)

        output = Dense(1, activation='sigmoid', name='gru_output')(h1)

        # model = Model(input=[input], output=[output])
        model = Model(input, output)

        return model

    def train(self, x_train, y_train, epochs, batch_size):
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)

    def save(self):
        model_json = self.model.to_json()
        with open("./saved_model/cnn.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("./saved_model/cnn.h5")
