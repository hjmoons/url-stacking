from tensorflow.keras.layers import Input, ELU, Embedding, BatchNormalization, Convolution1D, MaxPooling1D, concatenate, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

class cnn(Model):
    def __init__(self, max_len=80, emb_dim=32, max_vocab_len=128, W_reg=regularizers.l2(1e-4)):
        super(cnn, self).__init__(name='cnn_model')

        self.max_vocab_len = max_vocab_len
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.W_reg = W_reg

        # Embedding layer
        self.emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, embeddings_regularizer=W_reg)
        self.emb_drop = Dropout(0.2)

        self.h1 = Dense(1024)
        self.h2 = Dense(256)
        self.h3 = Dense(64)

        self.bn = BatchNormalization()
        self.el = ELU()
        self.dr = Dropout(0.5)

        # Output layer (last fully connected layer)
        # 마지막 클래스 결정하는 layer
        self.output_layer = Dense(1, activation='sigmoid', name='cnn_output')

    #def call(self, inputs=Input(shape=(80,), dtype='int32', name='cnn_input'), training=None, mask=None):
    def call(self, inputs, training=None, mask=None):
        print('##### input: ', inputs)
        x = self.emb(inputs)
        x = self.emb_drop(x)

        def get_conv_layer(emb, kernel_size=5, filters=256):

            def sum_1d(X):
                return K.sum(X, axis=1)

            # Conv layer
            conv = Convolution1D(kernel_size=kernel_size, filters=filters, padding='same')(emb)
            conv = ELU()(conv)
            conv = MaxPooling1D(5)(conv)
            conv = Lambda(sum_1d, output_shape=(filters,))(conv)
            conv = Dropout(0.5)(conv)
            
            return conv

        # Multiple Conv Layers
        # 커널 사이즈를 다르게 한 conv
        conv1 = get_conv_layer(x, kernel_size=2, filters=256)
        conv2 = get_conv_layer(x, kernel_size=3, filters=256)
        conv3 = get_conv_layer(x, kernel_size=4, filters=256)
        conv4 = get_conv_layer(x, kernel_size=5, filters=256)

        # Fully Connected Layers
        # 위 결과 합침
        merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

        hidden1 = self.h1(merged)
        hidden1 = self.el(hidden1)
        hidden1 = self.bn(hidden1)
        hidden1 = self.dr(hidden1)

        hidden2 = self.h2(hidden1)
        hidden2 = self.el(hidden2)
        hidden2 = self.bn(hidden2)
        hidden2 = self.dr(hidden2)

        hidden3 = self.h3(hidden2)
        hidden3 = self.el(hidden3)
        hidden3 = self.bn(hidden3)
        hidden3 = self.dr(hidden3)

        return self.output_layer(hidden3)


