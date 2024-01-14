import random

import tensorflow as tf
from keras import layers, Sequential
from keras.losses import Loss
from keras.src.backend import binary_crossentropy

from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet, AlternatedRealImaginaryDataSet
from Evaluation.plots import PredictedStepPlot
from Evaluation.statistic_errors import MSEEvaluateur, RMSEEvaluateur
import keras.backend as K
import keras
from sklearn.preprocessing import StandardScaler
from Models.BasicAutoEncoder import BasicAutoEncoder
import os

class CosineSimilarityLoss(Loss):
    def __init__(self, name='cosine_similarity_loss', **kwargs):
        super(CosineSimilarityLoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        # Récupérer les indices des deux plus grandes valeurs dans y_true et y_pred
        #indices_true = K.argsort(y_true, axis=-1)[:, -2:]
        #indices_pred = K.argsort(y_pred, axis=-1)[:, -2:]

        # Trier les indices
        #sorted_indices_true = K.sort(indices_true)
        #sorted_indices_pred = K.sort(indices_pred)

        # Calculer le RMSE entre les indices triés
        #rmse = K.sqrt(K.mean(K.square(sorted_indices_true - sorted_indices_pred), axis=-1))

        return tf.reduce_mean(tf.math.square(y_pred - y_true))
class BasicNNModel:

    def __init__(self, model = None):
        self.model = model


    class Trainer:
        def custom_loss(self, y_true, y_pred, threshold=0.4):
            # Calculer la distance angulaire entre les prédictions et les vraies étiquettes
            angular_distance = tf.abs(tf.math.subtract(y_pred, y_true))

            # Appliquer la perte binaire habituelle
            base_loss = binary_crossentropy(y_true, y_pred)

            # Appliquer une pénalité pour les erreurs d'un degré
            penalty = 0.1 * angular_distance

            # Combiner la perte binaire et la pénalité
            total_loss = base_loss + penalty

            return total_loss

        def custom_weighted_binary_crossentropy(self, y_true, y_pred):
            # Calculez la perte binaire cross-entropy standard
            binary_crossentropy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

            # Calculez les poids en fonction de la distance angulaire
            weights = tf.math.log1p(tf.abs(y_true - y_pred))

            # Appliquez les poids à la perte
            weighted_loss = tf.reduce_sum(binary_crossentropy_loss * weights) / tf.reduce_sum(weights)

            return weighted_loss
        def __init__(self, input_shape, output_dim):
            # Créez un modèle séquentiel
            self.input_shape = input_shape
            model = Sequential()
            self.model = model
            model.add(layers.Input(
                shape=input_shape))
            model.add(layers.Dense(150, activation='relu'))
            model.add(layers.Dense(150, activation='relu'))
            model.add(layers.Dense(150, activation='relu'))
            model.add(layers.Dense(150, activation='relu'))
            # Ajout de la couche de sortie avec 1 neurone (classification binaire) et une fonction d'activation sigmoïde
            model.add(layers.Dense(output_dim, activation='sigmoid'))

            # Compilez le modèle
            model.compile(optimizer='adam', loss=self.custom_loss, metrics=['accuracy'],run_eagerly=True)





        def train(self, X_train, y_train, epochs = 10, batch_size = 1, validation_data=None):
            # Entraînez le modèle avec vos données d'entraînement et vos étiquettes
            self.X_train = X_train
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
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
        self.model = keras.models.load_model(os.getcwd()+'/Models/saved/'+name)

if __name__ == "__main__":
    absolutePath = "C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Data/"
    data_loader = DataLoader([absolutePath+"Dataset_X8004_2S.csv",absolutePath+"Dataset_X2922_2S.csv",absolutePath+"Dataset_X9193_3-2S.csv",absolutePath+"Dataset_X4559_3-2S.csv"],
                             [absolutePath+"Dataset_y8004_2S.csv",absolutePath+"Dataset_y2922_2S.csv",absolutePath+"Dataset_y9193_3-2S.csv",absolutePath+"Dataset_y4559_3-2S.csv"])
    data, labels = data_loader.load_data()
    radar_dataset = AlternatedRealImaginaryDataSet(data, labels, 0.1, appended_snr=True)


    trainer2 = BasicNNModel.Trainer((200,), 181)
    trainer2.train(radar_dataset.X_train, radar_dataset.y_train, epochs=20, batch_size=500,validation_data=(radar_dataset.X_validation, radar_dataset.y_validation))
    trainer2.saveModel("tempDNNModel_4_1000")

