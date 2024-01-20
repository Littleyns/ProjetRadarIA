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
import pandas as pd
class RNNModel:
    def __init__(self, model = None):
        self.model = model
    class Trainer:
        def __init__(self):
            # Créez un modèle séquentiel
            self.model = Sequential()
            self.model.add(layers.LSTM(181, input_shape=(5, 181), activation='linear'))
            self.model.add(layers.Dense(181, activation="linear"))
            self.model.compile(optimizer='adam', loss='mean_squared_error')

            # Résumé du modèle
            self.model.summary()

        def train(
                self, X_train, y_train, epochs=30, batch_size=1, validation_split=0.2
        ):
            # Entraînez le modèle avec vos données d'entraînement et vos étiquettes
            self.X_train = X_train
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
            )
            # Affichez un résumé du modèle
            self.model.summary()
            return history

        def saveModel(self, name):
            self.model.save("./saved/" + name)

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
def get_dataset(path):
  data = pd.read_csv(path,header=None, index_col=False)
  X = data.iloc[1:,:5].values
  y = data.iloc[1:,5].values
  sy = list(map(lambda yy: yy.strip('][').split(', '),y))
  y = np.array(sy).astype(np.float32)
  X = np.array([ list(map(lambda mesure: mesure.strip('][').split(', '),X[i]))for i in range(len(X))]).astype(np.float32)
  return X,y
if __name__ == "__main__":
    X, y = get_dataset("C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Data/temporal_measures_trainset.csv")


    trainer = RNNModel.Trainer()
    history = trainer.train(X, y, epochs = 70, batch_size=50, validation_split=0.2)#, validation_data=(radar_dataset.X_test,radar_dataset.y_test))

    trainer.saveModel("RNNModel1")

