from tensorflow.keras.optimizers import Adam

from cnn import cnn
from model_preprocessor import Preprocessor

model = cnn()

x_train, x_test, y_train, y_test = Preprocessor.load_data_binary(10000)

# Define Deep Learning Model

''' Training phrase '''
epochs = 15
batch_size = 64
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.11)
