# Load Libraries
import warnings

from cnn import UrlCNN
from gru import UrlGRU
from lstm import UrlLSTM
from preprocessor import Preprocessor
warnings.filterwarnings("ignore")

epochs = 5
batch_size = 64

x_train, x_test, y_train, y_test = Preprocessor.load_data_binary(10000)

model = UrlLSTM()

model.train(x_train, y_train, epochs, batch_size)



