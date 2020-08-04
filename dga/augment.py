from eda import *
import numpy as np
import argparse
import tensorflow as tf
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented domains per original domain")
ap.add_argument("--alpha", required=False, type=float, help="percent of chars in each domain to be changed")
args = ap.parse_args()

num_aug = 4     # default
if args.num_aug:
    num_aug = args.num_aug


alpha = 0.1     # default
if args.alpha:
    alpha = args.alpha


def gen_eda(x_train, y_train, alpha=0.1, num_aug=4):

    x_train_aug = x_train
    y_train_aug = y_train
    y_train_aug = [np.argmax(y, axis=None, out=None) for y in y_train_aug]

    n = len(y_train)    # # of data

    for i in range(n):
        label = np.argmax(y_train[i])

        # bottom 5 classes
        if label >= 16:
            x_train_aug = np.append(x_train_aug, eda(x_train[i], alpha_rs=alpha, num_aug=num_aug), axis=0)
            for k in range(num_aug + 1):
                y_train_aug = np.append(y_train_aug, np.array(label))

    y_train_aug = tf.keras.utils.to_categorical(y_train_aug, 21)

    print('Before Augmentation: ', Counter(np.argmax(y_train, axis=1)))
    print('After Augmentation: ', Counter(np.argmax(y_train_aug, axis=1)))

    return x_train_aug, y_train_aug
