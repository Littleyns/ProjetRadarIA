import random

import tensorflow as tf
from keras import Sequential, layers
import keras


from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet
from Evaluation.plots import PredictedStepPlot, LearningCurvesPlot
from Evaluation.statistic_errors import MSEEvaluateur, RMSEEvaluateur
from sklearn.preprocessing import StandardScaler
from Models.BasicAutoEncoder import BasicAutoEncoder


class BasicCNNModel:
    def __init__(self, model = None):
        self.model = model
    class Trainer:
        def __init__(self, input_shape, output_dim):
            # Créez un modèle séquentiel
            self.input_shape = input_shape
            model = keras.Sequential()

            # Couche de convolution 1D avec 32 filtres, une fenêtre de 3 et une fonction d'activation ReLU
            model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_shape[1], 1)))

            # Couche de convolution 1D supplémentaire
            model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))

            # Aplatissement des données pour la couche dense
            model.add(layers.Flatten())

            # Couche dense (entièrement connectée) avec 128 neurones
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dense(64, activation='relu'))
            # Couche de sortie avec une seule unité pour la classification binaire (par exemple, sigmoid pour la classification binaire)
            model.add(layers.Dense(output_dim, activation='sigmoid'))

            # Compiler le modèle avec une fonction de perte (loss) appropriée et un optimiseur
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            # Résumé du modèle
            model.summary()
            self.model = model;





        def train(self, X_train, y_train, epochs = 30, batch_size = 1, validation_data = None):
            # Entraînez le modèle avec vos données d'entraînement et vos étiquettes
            self.X_train = X_train
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
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
        self.model = keras.models.load_model('C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Models/saved/'+name)

if __name__ == "__main__":

    data_loader = DataLoader()
    data, labels = data_loader.load_data()
    radar_dataset = RadarDataSet(data, labels, 0.05)

    trainer = BasicCNNModel.Trainer(radar_dataset.X_train.shape, 180)
    history = trainer.train(radar_dataset.X_train, radar_dataset.y_train, epochs = 30, batch_size=10, validation_data=(radar_dataset.X_test,radar_dataset.y_test))
    learningCurvePloter = LearningCurvesPlot()
    learningCurvePloter.evaluate(history)
    trainer.saveModel("basicCNNModel4_moreData")

