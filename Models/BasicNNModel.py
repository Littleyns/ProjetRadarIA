import random

import tensorflow as tf
from keras import Sequential
from keras import layers
import keras
from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet
from Evaluation.plots import PredictedStepPlot
from Evaluation.statistic_errors import MSEEvaluateur, RMSEEvaluateur
from sklearn.preprocessing import StandardScaler
from Models.BasicAutoEncoder import BasicAutoEncoder


class BasicNNModel:

    def __init__(self, model = None):
        self.model = model
    class Trainer:
        def __init__(self, input_shape, output_dim):
            # Créez un modèle séquentiel
            self.input_shape = input_shape
            model = Sequential()
            self.model = model
            # Ajoutez une couche de convolution 1D
            #model.add(Conv2D(32, kernel_size=2, activation='relu', input_shape=input_shape))
            # Ajoutez une autre couche de convolution 1D
            #model.add(Conv2D(64, kernel_size=2, activation='relu'))


            # Aplatissez les données pour les couches fully connected
            #model.add(Flatten())

            # Ajoutez une ou plusieurs couches fully connected pour la prédiction de la DOA
            model.add(layers.Input(
                shape=(input_shape[1],)))  # Remplacez X_train.shape[1] par la dimension de vos données d'entrée

            # Ajout de deux couches cachées avec 64 neurones et une fonction d'activation ReLU
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.Dense(1000, activation='relu'))
            # Ajout de la couche de sortie avec 1 neurone (classification binaire) et une fonction d'activation sigmoïde
            model.add(layers.Dense(output_dim, activation='sigmoid'))

            # Compilez le modèle
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])





        def train(self, X_train, y_train, epochs = 10, batch_size = 1):
            # Entraînez le modèle avec vos données d'entraînement et vos étiquettes
            self.X_train = X_train
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
            # Affichez un résumé du modèle
            self.model.summary()
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
        self.model = keras.models.load_model('C:/Users/HP/Desktop/rad/Models/saved/'+name)

if __name__ == "__main__":
    data_loader = DataLoader()
    data, labels = data_loader.load_data()
    radar_dataset = RadarDataSet(data, labels, 0.1)


    #USING Autoencoded complex number
    basicAutoEncoder = BasicAutoEncoder()
    basicAutoEncoder.load("basicAutoEncoder")
    X_train_encoded = basicAutoEncoder.encode(radar_dataset.X_train).squeeze()

    #trainer = BasicNNModel.Trainer(radar_dataset.X_train.shape, 180)
    #trainer.train(radar_dataset.X_train, radar_dataset.y_train, 50,2)
    #trainer.saveModel("basicNNModel1")


    trainer2 = BasicNNModel.Trainer(radar_dataset.X_train.shape, 180)
    trainer2.train(radar_dataset.X_train, radar_dataset.y_train, 50, 2)
    trainer2.saveModel("tempDNNModel_4_1000")

