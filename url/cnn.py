# Load Libraries
from keras import backend as K
from keras import regularizers
from keras.layers import Input, ELU, Embedding, BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.layers.core import Dense, Dropout, Lambda
from keras.models import Model


def define_model(max_len=80, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):
    """CNN model with the Keras functional API"""

    # Input
    main_input = Input(shape=(max_len,), dtype='int32', name='cnn_input')

    # Embedding layer
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg)(main_input)
    emb = Dropout(0.2)(emb)

    def sum_1d(X):
        return K.sum(X, axis=1)

    def get_conv_layer(emb, kernel_size=5, filters=256):
        # Conv layer
        conv = Convolution1D(kernel_size=kernel_size, filters=filters, border_mode='same')(emb)
        conv = ELU()(conv)
        conv = MaxPooling1D(5)(conv)
        conv = Lambda(sum_1d, output_shape=(filters,))(conv)
        conv = Dropout(0.5)(conv)
        return conv

    # Multiple Conv Layers
    # 커널 사이즈를 다르게 한 conv
    conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
    conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
    conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
    conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

    # Fully Connected Layers
    # 위 결과 합침
    merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

    hidden1 = Dense(1024)(merged)
    hidden1 = ELU()(hidden1)
    hidden1 = BatchNormalization(mode=0)(hidden1)
    hidden1 = Dropout(0.5)(hidden1)

    hidden2 = Dense(256)(hidden1)
    hidden2 = ELU()(hidden2)
    hidden2 = BatchNormalization(mode=0)(hidden2)
    hidden2 = Dropout(0.5)(hidden2)

    hidden3 = Dense(64)(hidden2)
    hidden3 = ELU()(hidden3)
    hidden3 = BatchNormalization(mode=0)(hidden3)
    hidden3 = Dropout(0.5)(hidden3)

    # Output layer (last fully connected layer)
    # 마지막 클래스 결정하는 layer
    output = Dense(1, activation='sigmoid', name='cnn_output')(hidden3)

    model = Model(input=[main_input], output=[output])

    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    return model
