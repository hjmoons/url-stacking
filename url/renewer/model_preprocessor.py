import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn import model_selection
from string import printable
from keras.preprocessing import sequence

class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def load_data_binary(data_num):
        """ Load and pre-process data.

        1) Load data from dir
        2) Tokenizing
        3) Padding

        return train and test data

        """
        train_num = int(data_num*0.8)
        test_num = int(data_num*0.2)
        train_half = int(train_num/2)
        
        # Load data
        data_home = 'data/'
        df = pd.read_csv(data_home + 'url_label.csv', encoding='ISO-8859-1', sep=',')
        
        class_0 = df['class'] == 0
        class_1 = df['class'] != 0
        
        class_0_df = df[class_0].sample(int(data_num/2))
        class_1_df = df[class_1].sample(int(data_num/2))
        
        print(class_1_df)
        
        train_df = pd.concat([class_0_df[:train_half], class_1_df[:train_half]])
        test_df = pd.concat([class_0_df[train_half:], class_1_df[train_half:]])
        
        def preprocessing(df, data_num):
            # Extract sample data 50000
            #df = df.sample(n=data_num)
            # Tokenizing domain string on character level
            # domain string to vector
            url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]

            # Padding domain integer max_len=64
            # 최대길이 80으로 지정
            max_len = 80
            x = sequence.pad_sequences(url_int_tokens, maxlen=max_len)

            # Lable data
            label_arr = []
            for i in df['class']:
                if i == 0:
                    label_arr.append(0)
                else :
                    label_arr.append(1)

            y = np.array(label_arr)

            x_data = pd.DataFrame(x.reshape(data_num, 80))
            y_data = pd.DataFrame(y.reshape(data_num, 1))

            return x_data, y_data
        
        x_train, y_train = preprocessing(train_df, train_num)
        x_test, y_test = preprocessing(test_df, test_num)
        
        print("x_train: ", x_train.shape)
        print("y_train: ", y_train.shape)
        print("x_test: ", x_test.shape)
        print("y_test: ", y_test.shape)
        
        return x_train, x_test, y_train, y_test