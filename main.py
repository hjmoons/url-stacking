# Load Libraries
import warnings

from preprocessor import Preprocessor
from stacking import UrlStacking

warnings.filterwarnings("ignore")

stack_model = UrlStacking()

x_train, x_test, y_train, y_test = Preprocessor.load_data_binary(10000)

stack_model.train(x_train, y_train, x_test, y_test, 4)


