import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn import model_selection

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils


def load_data_sample():
    """ Load and pre-process data.

    1) Load data from dir
    2) Tokenizing
    3) Padding

    return train and test data

    """

    # Load data
    data_home = 'data/'
    df = pd.read_csv(data_home + 'dga_label.csv', encoding='ISO-8859-1', sep=',')

    data = pd.DataFrame(columns=['domain', 'source', 'class'])

    for i in range(0, 21):
        class_ = df['class'] == i
        data = pd.concat([data, df[class_].sample(2000)])

    # Tokenizing domain string on character level
    # domain string to vector
    tokenizer = Tokenizer(filters='', lower=True, char_level=True)
    tokenizer.fit_on_texts(data.domain)
    url_int_tokens = tokenizer.texts_to_sequences(data.domain)

    # Padding domain integer max_len=64
    # 최대길이 73으로 지정
    max_len = 74

    x = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    y = np.array(data['class'])

    # Cross-validation
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=33)

    # DGA class: 0~20: 21개
    y_train_category = np_utils.to_categorical(y_train, 21)
    y_test_category = np_utils.to_categorical(y_test, 21)

    print("* * * Train Set * * *")
    print("x_train's shape: ", x_train.shape)
    print("y_train's shape: ", y_train_category.shape)
    print()
    print("* * * Test  Set * * *")
    print("x_test's shape: ", x_test.shape)
    print("y_test's shape: ", y_test_category.shape)
    print()

    return x_train, x_test, y_train_category, y_test_category


def load_data(data_path):
    """ Load and pre-process data.
    1) Load data from dir
    2) Tokenizing
    3) Padding
    return train and test data
    """

    # Load data
    df = pd.read_csv(data_path, encoding='ISO-8859-1', sep=',')

    # Tokenizing domain string on character level
    # domain string to vector
    tokenizer = Tokenizer(filters='', lower=True, char_level=True)
    tokenizer.fit_on_texts(df.domain)
    url_int_tokens = tokenizer.texts_to_sequences(df.domain)

    # Padding domain integer max_len=64
    # 최대길이 73으로 지정
    max_len = 73

    x = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
    y = np.array(df['class'])

    # Cross-validation
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=33)

    # DGA class: 0~20: 21개
    y_train_category = np_utils.to_categorical(y_train, 21)
    y_test_category = np_utils.to_categorical(y_test, 21)

    return x_train, x_test, y_train_category, y_test_category