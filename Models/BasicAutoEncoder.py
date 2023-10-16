import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet
from PreProcessing.utils import data_to_complex


class BasicAutoEncoder:
    def __init__(self, model = None):
        self.model = model
    class Trainer:
        def __init__(self, input_shape, encoded_dim):
            self.input_shape = input_shape
            self.encoded_dim = encoded_dim
            input_layer = Input(shape=input_shape)
            flatten_layer = tf.keras.layers.Flatten()(input_layer)  # Aplatir les données
            encoded_layer = Dense(encoded_dim, activation='relu')(flatten_layer)  # Couche de codage
            decoded_layer = Dense(100, activation='sigmoid')(encoded_layer)  # Couche de décodage
            reshape_layer = tf.keras.layers.Reshape(target_shape=(1,-1))(decoded_layer) # Remodeler les données décodées

            # Créer le modèle
            self.autoencoder = Model(input_layer, reshape_layer)

            # Compiler le modèle
            self.autoencoder.compile(optimizer='adam', loss='mse')

        def train(self, X_train):
            dataComplex = data_to_complex(X_train) #matrice (nombreDonnees, 2 ,100)

            self.autoencoder.fit(dataComplex, dataComplex, epochs=70, batch_size=2)
        def saveModel(self, name):
            self.autoencoder.save("./saved/"+name)
    def encode(self, data):
        dataComplex = data_to_complex(data)  # matrice (nombreDonnees, 2 ,100)
        return self.autoencoder.predict(dataComplex)
    def load(self, name):
        self.autoencoder = tf.keras.models.load_model('C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Models/saved/' + name)


if __name__ == "__main__":
    data_loader = DataLoader()
    data, labels = data_loader.load_data()
    radar_dataset = RadarDataSet(data, labels, 0.1)

    basicAutoEncoderTrainer = BasicAutoEncoder.Trainer((2, 100), 10)
    basicAutoEncoderTrainer.train(radar_dataset.X_train)
    basicAutoEncoderTrainer.saveModel("basicAutoEncoder")