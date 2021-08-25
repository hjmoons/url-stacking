# Load Libraries
import warnings

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, ELU, Embedding, BatchNormalization, Convolution1D, MaxPooling1D, concatenate, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


warnings.filterwarnings("ignore")


class UrlCNN:
    def __init__(self):
        self.model = self.cnn()

    @staticmethod
    def cnn(max_len=80, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):
        # Input
        input = Input(shape=(max_len,), dtype='int32', name='cnn_input')

        # Embedding layer
        #
        # URL을 학습하도록 문자를 숫자 데이터로 임베딩
        emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len,
                        embeddings_regularizer=W_reg)(input)
        emb = Dropout(0.2)(emb)

        def sum_1d(X):
            return K.sum(X, axis=1)

        def get_conv_layer(emb, kernel_size=5, filters=256):
            # Conv layer
            conv = Convolution1D(kernel_size=kernel_size, filters=filters, padding='same')(emb)
            conv = ELU()(conv)
            conv = MaxPooling1D(5)(conv)
            conv = Lambda(sum_1d, output_shape=(filters,))(conv)
            conv = Dropout(0.5)(conv)
            return conv

        # Multiple Conv Layers
        #
        # URL 특성 추출 단계
        # Convolution Layer의 커널 사이즈를 각각 다르게 설정
        conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
        conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
        conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
        conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

        # 4개의 Convolution Layer에서 나온 특성 값을 Concatenate Layer를 통해 결과 합침
        merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        # Fully Connected Layers
        #
        # URL 분류 단계
        # 추출된 특성에 따라 악성인지 아닌지 판단
        hidden1 = Dense(1024)(merged)
        hidden1 = ELU()(hidden1)
        hidden1 = BatchNormalization()(hidden1)
        hidden1 = Dropout(0.5)(hidden1)

        hidden2 = Dense(256)(hidden1)
        hidden2 = ELU()(hidden2)
        hidden2 = BatchNormalization()(hidden2)
        hidden2 = Dropout(0.5)(hidden2)

        hidden3 = Dense(64)(hidden2)
        hidden3 = ELU()(hidden3)
        hidden3 = BatchNormalization()(hidden3)
        hidden3 = Dropout(0.5)(hidden3)

        # 마지막 클래스 결정하는 Output layer
        output = Dense(1, activation='sigmoid', name='cnn_output')(hidden3)

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
