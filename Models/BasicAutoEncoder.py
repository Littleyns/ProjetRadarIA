import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet
from PreProcessing.utils import data_to_complex
import os

class BasicAutoEncoder:
    def __init__(self, model = None):
        self.model = model
    class Trainer:
        def __init__(self):

            input_layer = Input(shape=(2, 10, 10))
            flatten_layer = tf.keras.layers.Flatten()(input_layer)  # Aplatir les données
            encoded_layer = Dense(50, activation='relu')(flatten_layer)  # Couche de codage
            decoded_layer = Dense(100, activation='sigmoid')(encoded_layer)  # Couche de décodage
            reshape_layer = tf.keras.layers.Reshape((1,10,10))(decoded_layer) # Remodeler les données décodées

            # Créer le modèle
            self.autoencoder = Model(input_layer, reshape_layer)

            # Compiler le modèle
            self.autoencoder.compile(optimizer='adam', loss='mse')

        def train(self, X_train):

            self.autoencoder.fit(X_train, X_train, epochs=30, batch_size=10)
        def saveModel(self, name):
            self.autoencoder.save("./saved/"+name)
    def encode(self, data):
        return self.autoencoder.predict(data)
    def load(self, name):
        self.autoencoder = tf.keras.models.load_model(os.getcwd()+'/saved/'+name)

if __name__ == "__main__":
    data_loader = DataLoader("C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Data/Dataset_X5585_SNR-3030.csv","C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Data/Dataset_y5585_SNR-3030.csv")
    data, labels = data_loader.load_data()

    #Autoencode real and im
    real_part = data[:, : data.shape[1] // 2].reshape(-1, 10, 10)
    imag_part = data[:, data.shape[1] // 2:].reshape(-1, 10, 10)
    Rxx_im_complex = np.stack((real_part, imag_part), axis=1)#(2,10,10)
    bae = BasicAutoEncoder()
    bae_trainer = BasicAutoEncoder.Trainer()
    bae_trainer.train(Rxx_im_complex)
    bae_trainer.saveModel("AutoEncoderRxx")
