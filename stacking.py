# Load Libraries
import warnings
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from cnn import UrlCNN
from gru import UrlGRU
from lstm import UrlLSTM

warnings.filterwarnings("ignore")

class UrlStacking:
    def __init__(self):
        self.cnn = UrlCNN()
        self.lstm = UrlLSTM()
        self.gru = UrlGRU()
        self.stack_model = self.model()
        self.acc = []

    @staticmethod
    def model():
        input = Input(shape=(3,), dtype='int32', name='cnn_input')
        hidden = Dense(10, activation='relu')(input)
        output = Dense(1, activation='sigmoid', name='final_output')(hidden)
        model = Model(input, output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def get_stacking_data(model, x_train, y_train, x_test, n_splits):
        kfold = KFold(n_splits=n_splits)

        train_fold_predict = np.zeros((x_train.shape[0], 1))
        test_predict = np.zeros((x_test.shape[0], n_splits))
        print("model: ", model.summary())
        for cnt, (train_index, valid_index) in enumerate(kfold.split(x_train)):
            x_train_ = x_train.loc[train_index]
            y_train_ = y_train.loc[train_index]
            x_validation = x_train.loc[valid_index]

            model.fit(x_train_, y_train_)

            train_fold_predict[valid_index, :] = model.predict(x_validation).reshape(-1, 1)

            test_predict[:, cnt] = model.predict(x_test).reshape(-1,)

        test_predict_mean = np.mean(test_predict, axis=1).reshape(-1, 1)
        print()
        return train_fold_predict, test_predict_mean

    def train(self, x_train, y_train, x_test, y_test, n_splits):
        cnn_train, cnn_test = self.get_stacking_data(self.cnn.model, x_train, y_train, x_test, n_splits)
        lstm_train, lstm_test = self.get_stacking_data(self.lstm.model, x_train, y_train, x_test, n_splits)
        gru_train, gru_test = self.get_stacking_data(self.gru.model, x_train, y_train, x_test, n_splits)

        stack_final_x_train = np.concatenate((cnn_train, lstm_train, gru_train), axis=1)
        stack_final_x_test = np.concatenate((cnn_test, lstm_test, gru_test), axis=1)
        #stack_final_x_train = np.concatenate((cnn_train, lstm_train), axis=1)
        #stack_final_x_test = np.concatenate((cnn_test, lstm_test), axis=1)

        self.stack_model.fit(stack_final_x_train, y_train, epochs=15, batch_size=64)
        stack_final = self.stack_model(stack_final_x_test)

        #print('stack model accuracy: {0:.4f}'.format(accuracy_score(y_test, stack_final)))
