# Load Libraries
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import concatenate


class UrlStacking:
    def __init__(self):
        pass

    @staticmethod
    def define_stacked_model(models):

        # update all layers in all models to not be trainable
        for i in range(len(models)):
            model = models[i]
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unique layer name' issue
                layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
        # define multi-headed input
        ensemble_visible = [model.input for model in models]
        # concatenate merge output from each model
        ensemble_outputs = [model.output for model in models]

        merge = concatenate(ensemble_outputs, name='final_input')
        hidden = Dense(10, activation='relu')(merge)
        output = Dense(1, activation='sigmoid', name='final_output')(hidden)
        model = Model(inputs=ensemble_visible, outputs=output)
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model

    # fit a stacked model
    @staticmethod
    def fit_stacked_model(model, inputX, inputy):
        # prepare input data
        X = [inputX for _ in range(len(model.input))]
        # encode output data

        # fit model
        model.fit(X, inputy, epochs=10, batch_size=64)

    # make a prediction with a stacked model
    @staticmethod
    def predict_stacked_model(model, inputX):
        # prepare input data
        X = [inputX for _ in range(len(model.input))]
        # make prediction
        return model.predict(X, verbose=0)

