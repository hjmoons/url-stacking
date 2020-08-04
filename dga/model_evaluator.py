import time
from datetime import datetime
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy import interp

class Evaluator:
    pass

    @staticmethod
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def fbeta_score(y_true, y_pred, beta=1):
        if beta < 0:
            raise ValueError('The lowest choosable beta is zero (only precision).')

        # If there are no true positives, fix the F score at 0 like sklearn.
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = Evaluator.precision(y_true, y_pred)
        r = Evaluator.recall(y_true, y_pred)

        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score

    @staticmethod
    def fmeasure(y_true, y_pred):
        return Evaluator.fbeta_score(y_true, y_pred, beta=1)

    @staticmethod
    def plot_validation_curves(model_name, history):
        """Save validation curves(.png format) """

        history_dict = history.history

        # validation curves
        epochs = range(1, len(history_dict['loss']) + 1)
        # "bo" is for "blue dot"
        plt.plot(epochs, history_dict['val_fmeasure'], 'r',label='F1')
        # plt.plot(epochs, history_dict['val_precision'], 'g',label='precision')
        # plt.plot(epochs, history_dict['val_recall'], 'k',label='recall')
        plt.plot(epochs, history_dict['val_loss'], 'k', label='loss')
        plt.plot(epochs, history_dict['val_categorical_accuracy'], 'c', label='categorical_accuracy')

        plt.xlabel('Epochs')
        plt.grid()
        plt.legend(loc='lower right')
        #plt.show()
        now = datetime.now()
        now_datetime = now.strftime('%Y_%m_%d-%H%M%S')

        plt.savefig('./result/' + model_name + '_val_curve_' + now_datetime + '.png')

        plt.close()

        # plt.plot(epochs, history_dict['val_f1_score_weighted'], 'r',label='F1_Weighted')
        # plt.plot(epochs, history_dict['val_f1_score_micro'], 'g',label='F1_Micro')
        # plt.plot(epochs, history_dict['val_f1_score_macro'], 'k',label='F1_Macro')
        #
        # plt.xlabel('Epochs')
        # plt.grid()
        # plt.legend(loc='lower right')
        # #plt.show()
        # time.sleep(5)
        # now = datetime.now()
        # now_datetime = now.strftime('%Y_%m_%d-%H%M%S')
        # plt.savefig('./result/' + model_name + '_val_curve_' + now_datetime + '.png')

    def print_validation_report(history):
            """Print validation history """
            history_dict = history.history

            for key in history_dict:
                if "val" in key:
                    print('[' + key + '] '+ str(history_dict[key]))

    @staticmethod
    def calculate_measure(model, x_test, y_test):
        """Calculate measure(categorical accuracy, precision, recall, F1-score) """

        y_pred_class_prob = model.predict(x_test, batch_size=64)
        y_pred_class = np.argmax(y_pred_class_prob, axis=1)
        y_true_class = np.argmax(y_test, axis=1)

        # classification report(sklearn)
        print(classification_report(y_true_class, y_pred_class, digits=4,
                                    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))

        print("weighted_precision" , metrics.precision_score(y_true_class, y_pred_class, average='weighted'))
        print("weighted_recall" , metrics.recall_score(y_true_class, y_pred_class, average='weighted'))
        print("weighted_F1" , metrics.f1_score(y_true_class, y_pred_class, average='weighted'))

        print("micro_precision" , metrics.precision_score(y_true_class, y_pred_class, average='micro'))
        print("micro_recall" , metrics.recall_score(y_true_class, y_pred_class, average='micro'))
        print("micro_F1" , metrics.f1_score(y_true_class, y_pred_class, average='micro'))

        print("macro_precision" , metrics.precision_score(y_true_class, y_pred_class, average='macro'))
        print("macro_recall" , metrics.recall_score(y_true_class, y_pred_class, average='macro'))
        print("macro_F1" , metrics.f1_score(y_true_class, y_pred_class, average='macro'))

    @staticmethod
    def plot_confusion_matrix(model_name, y_true, y_pred,
                              classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                              normalize=True,
                              title=None,
                              cmap=plt.cm.Blues):
        """Save confusion matrix(.png) """

        dga_labels_dict = {'Alexa':0, 'banjori':1, 'qsnatch':2, 'tinba':3, 'Post':4, 'ramnit':5, 'qakbot':6,
                           'necurs':7, 'murofet':8, 'shiotob/urlzone/bebloh':9, 'monerodownloader':10, 'simda':11,
                           'ranbyus':12, 'pykspa':13, 'kraken':14, 'dyre':15 , 'nymaim':16, 'Cryptolocker': 17, 'locky': 18,
                           'vawtrak':19, 'qadars':20}
        classes_str = []
        for i in classes:
            for dga_str, dga_int in dga_labels_dict.items():
                if dga_int == i:
                    classes_str.append(dga_str)

        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_true, axis=1)

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true_class, y_pred_class)
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        # Only use the labels that appear in the data
        #classes = list(classes[unique_labels(y_true, y_pred)])

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes_str, yticklabels=classes_str,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        precision = metrics.precision_score(y_true_class, y_pred_class, average = 'weighted')
        recall = metrics.recall_score(y_true_class, y_pred_class, average = 'weighted')
        F1 = metrics.f1_score(y_true_class, y_pred_class, average = 'weighted')

        plt.xlabel('Predicted label\naccuracy={:0.4f}; precision={:0.4f}; recall={:0.4f}; F1={:0.4f}; misclass={:0.4f}'
                   .format(accuracy, precision, recall, F1, misclass))
        now = datetime.now()
        now_datetime = now.strftime('%Y_%m_%d-%H%M%S')
        figure = plt.gcf()
        figure.set_size_inches(15, 15)
        plt.savefig('./result/' + model_name + '_confusion_matrix_' + now_datetime + '.png', dpi=100)
        plt.close()
        plt.cla()

    @staticmethod
    def plot_roc_curves(model_name, y_true, y_pred):

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = 21

        # Calculate fpr , tpr, and roc(each class, micro, macro)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        colors = cycle(['blue', 'red', 'green'])

        for i, color in zip(range(3), colors):
            plt.plot(fpr[i], tpr[i], color=color, label='ROC curve of class {0} (AUC = {1:0.4f})' ''.format(i, roc_auc[i]))

        for i, color in zip(range(18, 21), cycle(['c', 'm', 'y'])):
            plt.plot(fpr[i], tpr[i], linestyle='-.', color=color, label='ROC curve of class {0} (AUC = {1:0.4f})' ''.format(i, roc_auc[i]))

        plt.plot(fpr["micro"], tpr["micro"], linestyle='--',
                 label='ROC curve of micro-average (AUC = {0:0.4f})' ''.format(roc_auc["micro"]))

        plt.plot(fpr["macro"], tpr["macro"], linestyle='--',
                 label='ROC curve of macro-average (AUC = {0:0.4f})' ''.format(roc_auc["macro"]))

        plt.plot([0, 1], [0, 1], 'k--')   # base line
        plt.xlim([0.00001, 1.0])
        plt.xscale('log')
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic(ROC Curve)')
        ax = plt.subplot()
        plt.legend(loc="right", bbox_to_anchor=(2, 0.2))
        plt.tight_layout()
        plt.subplots_adjust(right=0.7)
        # plt.show()
        now = datetime.now()
        now_datetime = now.strftime('%Y_%m_%d-%H%M%S')
        plt.savefig('./result/' + model_name + '_roc_curve_' + now_datetime + '.png', dpi=500, bbox_inches="tight")