import random
import os
import tensorflow as tf
from keras import Sequential, layers
import keras
from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet
from Evaluation.plots import PredictedStepPlot, LearningCurvesPlot
from Evaluation.statistic_errors import MSEEvaluateur, RMSEEvaluateur
from sklearn.preprocessing import StandardScaler
from Models.BasicAutoEncoder import BasicAutoEncoder
import numpy as np

class RNNModel:
    def __init__(self, model = None):
        self.model = model
    class Trainer:
        def __init__(self, input_shape, output_dim):
            # Créez un modèle séquentiel
            self.model = Sequential()
            model.add(layers.InputLayer(input_shape=(1, input_shape[1], 1)))
            self.model.add(layers.ConvLSTM1D(filters=64, kernel_size=(1)))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(output_dim))
            self.model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy','mae', 'mse'])

            # Résumé du modèle
            self.model.summary()





        def train(self, X_train, y_train, epochs = 30, batch_size = 10, validation_data = None):
            # Entraînez le modèle avec vos données d'entraînement et vos étiquettes
            self.X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],1)
            history = self.model.fit(self.X_train, y_train, epochs=epochs, batch_size=batch_size)#, validation_data=validation_data)
            # Affichez un résumé du modèle
            self.model.summary()
            return history
        def saveModel(self, name):
            self.model.save("./saved/"+name)

    def evaluate(self, X_test, y_test):
        tf.config.run_functions_eagerly(True)
        test_loss = self.model.evaluate(X_test, y_test)
        print("Loss evaluation: "+ str(test_loss))
        y_predicted = self.model.predict(X_test)
        MSEEvaluateur().evaluate(y_test, y_predicted)
        RMSEEvaluateur().evaluate(y_test, y_predicted)

        randomTestIndex = random.randint(0, len(y_predicted))
        PredictedStepPlot().evaluate(y_test[randomTestIndex], y_predicted[randomTestIndex])

    def predict(self, test_data):
        return self.model.predict(test_data)

    def load(self, name):
        self.model = keras.models.load_model(os.getcwd()+'/Models/saved/'+name)

if __name__ == "__main__":
    data_loader = DataLoader("C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Data/Dataset_X6687.csv","C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Data/Dataset_y6687.csv")
    data, labels = data_loader.load_data()
    radar_dataset = RadarDataSet(data, labels, 0.05)

    trainer = RNNModel.Trainer(radar_dataset.X_train.shape, 180)
    history = trainer.train(radar_dataset.X_train, radar_dataset.y_train, epochs = 30, batch_size=10)#, validation_data=(radar_dataset.X_test,radar_dataset.y_test))
    learningCurvePloter = LearningCurvesPlot()
    learningCurvePloter.evaluate(history)
    trainer.saveModel("RNNModel1")

