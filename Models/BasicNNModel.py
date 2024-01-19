import random

import tensorflow as tf
from keras import layers, Sequential
from keras.losses import Loss
from keras.src.backend import binary_crossentropy

from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet, AlternatedRealImaginaryDataSet, RealImaginaryXDataSet
from Evaluation.plots import PredictedStepPlot
from Evaluation.statistic_errors import MSEEvaluateur, RMSEEvaluateur
import keras.backend as K
import keras
from sklearn.preprocessing import StandardScaler
from Models.BasicAutoEncoder import BasicAutoEncoder
import os


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

        @staticmethod
        def angleSensitiveCustomLoss(y_true, y_pred):
            # Trouver les indices des occurrences de 1 dans y_true pour chaque exemple du batch
            angles = tf.where(tf.equal(y_true, 1))
            angles_penalties = tf.TensorArray(tf.float32, size=tf.shape(y_true)[0], dynamic_size=True,
                                              clear_after_read=False)

            def loop_body(i, angles_penalties):
                indices_ones_i = tf.where(tf.equal(y_true[i], 1))
                theta = tf.range(0, 181, 1, dtype=tf.float32)
                penalties_raw = tf.abs(tf.cast(indices_ones_i, tf.float32) - theta)
                penalties = tf.reduce_min(penalties_raw, axis=0) * y_pred[i]
                angles_penalties.write(i, penalties).mark_used()

                return i + 1, angles_penalties

            _, angles_penalties = tf.while_loop(
                lambda i, _: i < tf.shape(y_true)[0],
                loop_body,
                [0, angles_penalties]
            )
            # Convertir angles_penalties en un Tensor
            angles_penalties = angles_penalties.stack()
            # Réduire les dimensions pour rendre les formes compatibles
            angles_penalties = tf.cast(tf.reduce_mean(tf.reduce_sum(angles_penalties, axis=1)), tf.float32)
            bce = tf.keras.losses.BinaryCrossentropy(axis=1)

            # Calculer la perte
            base_loss = bce(tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32))
            # Utiliser angles_penalties dans le calcul de la perte
            loss = base_loss + (0.05 * angles_penalties)

            return loss
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
            model.compile(optimizer='adam', loss=self.angleSensitiveCustomLoss, metrics=['accuracy'],run_eagerly=True)





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
    data_loader = DataLoader([absolutePath + "Dataset_X165_2S.csv", absolutePath + "Dataset_X7979_3-2S.csv",
                              absolutePath + "Dataset_X8004_2S.csv", absolutePath + "Dataset_X2922_2S.csv",
                              absolutePath + "Dataset_X9193_3-2S.csv", absolutePath + "Dataset_X4559_3-2S.csv",
                              absolutePath + "Dataset_X2599_3-2S.csv", absolutePath + "Dataset_X4523_3S.csv"],
                             [absolutePath + "Dataset_y165_2S.csv", absolutePath + "Dataset_y7979_3-2S.csv",
                              absolutePath + "Dataset_y8004_2S.csv", absolutePath + "Dataset_y2922_2S.csv",
                              absolutePath + "Dataset_y9193_3-2S.csv", absolutePath + "Dataset_y4559_3-2S.csv",
                              absolutePath + "Dataset_y2599_3-2S.csv", absolutePath + "Dataset_y4523_3S.csv"])
    data, labels = data_loader.load_data()
    radar_dataset = RadarDataSet(data, labels, 0.1, appended_snr=True)


    trainer2 = BasicNNModel.Trainer((200,), 181)
    trainer2.train(radar_dataset.X_train, radar_dataset.y_train, epochs=10, batch_size=2000,validation_data=(radar_dataset.X_validation, radar_dataset.y_validation))
    trainer2.saveModel("DNN1_e10,b2000_sensitiveAngleLoss_Alternate_concatenatedXRI")

