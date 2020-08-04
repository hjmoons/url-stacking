import warnings

import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, ELU, Embedding, BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.layers.core import Dense, Dropout, Lambda
from keras.models import Model
from keras.optimizers import Adam
from model_evaluator import Evaluator
from model_preprocessor import Preprocessor

# GPU memory setting
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
K.tensorflow_backend.set_session(tf.Session(config=config))

warnings.filterwarnings('ignore')

def conv_fully(max_len=74, emb_dim=32, max_vocab_len=40, W_reg=regularizers.l2(1e-4)):
    """CNN model with the Keras functional API"""

    # Input
    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')

    # Embedding layer
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, input_length=max_len, W_regularizer=W_reg)(main_input)
    emb = Dropout(0.2)(emb)

    def get_conv_layer(emb, kernel_size=5, filters=256):
        # Conv layer
        conv = Convolution1D(kernel_size=kernel_size, filters=filters, padding='same')(emb)
        conv = ELU()(conv)
        conv = MaxPooling1D(5)(conv)
        conv = Lambda(lambda x: K.sum(x, axis=1), output_shape=(filters,))(conv)
        conv = Dropout(0.5)(conv)

        return conv

    conv1 = get_conv_layer(emb, kernel_size=2, filters=256)
    conv2 = get_conv_layer(emb, kernel_size=3, filters=256)
    conv3 = get_conv_layer(emb, kernel_size=4, filters=256)
    conv4 = get_conv_layer(emb, kernel_size=5, filters=256)

    merged = concatenate([conv1, conv2, conv3, conv4], axis=1)

    hidden1 = Dense(1024, activation='relu')(merged)
    hidden1 = ELU()(hidden1)
    hidden1 = BatchNormalization()(hidden1)
    hidden1 = Dropout(0.5)(hidden1)

    hidden2 = Dense(1024, activation='relu')(hidden1)
    hidden2 = ELU()(hidden2)
    hidden2 = BatchNormalization()(hidden2)
    hidden2 = Dropout(0.5)(hidden2)

    main_output = Dense(21, activation='softmax')(hidden2)

    cnn_model = Model(inputs=main_input, outputs=main_output)

    return cnn_model


x_train, x_test, y_train, y_test = Preprocessor.load_data()

model_name = "CNN"
model = conv_fully()
model.summary()

# Plot model(.png)
# tf.keras.utils.plot_model(model, to_file="./result/" + model_name)

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