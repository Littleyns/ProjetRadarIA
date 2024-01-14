import random

import tensorflow as tf
from keras import Sequential, layers
import keras
import os
import numpy as np
from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet, RealImaginaryRxxDataSet, RealImaginaryXDataSet
from Evaluation.plots import PredictedStepPlot, LearningCurvesPlot
from Evaluation.statistic_errors import MSEEvaluateur, RMSEEvaluateur, R2Score, Accuracy
from sklearn.preprocessing import StandardScaler
from Models.BasicAutoEncoder import BasicAutoEncoder
from PreProcessing.utils import data_to_complex
import keras.backend as K
from keras.backend import binary_crossentropy
class DocuCNNModel:
    def __init__(self, model=None):
        self.model = model

    class Trainer:
        def leaky_relu(x, alpha=0.01):
            return tf.maximum(alpha * x, x)
        @staticmethod
        def custom_loss(y_true, y_pred, threshold=0.4):
            # Calculer la distance angulaire entre les prédictions et les vraies étiquettes
            angular_distance = tf.abs(tf.math.subtract(y_pred, y_true))
            print("----------------")
            print(angular_distance)
            # Appliquer la perte binaire habituelle
            base_loss = binary_crossentropy(y_true, y_pred)

            # Appliquer une pénalité pour les erreurs d'un degré
            penalty = 0.5 * angular_distance
            # Combiner la perte binaire et la pénalité
            total_loss = base_loss + penalty

            return total_loss

        @staticmethod
        def klDivergenceLoss(y_true,y_pred, alpha=0.1):
                # Calculer la divergence de Kullback-Leibler
            kl_divergence = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
            # Appliquer une pondération en fonction de la classe réelle
            kl_divergence = tf.expand_dims(kl_divergence, axis=-1)
            weighted_loss = tf.reduce_sum(kl_divergence * (1.0 - y_true), axis=1)

            # Appliquer une pénalité moindre pour les erreurs proches de la classe réelle
            total_loss = tf.reduce_mean(weighted_loss) + alpha * tf.reduce_mean(kl_divergence)

            return total_loss

        def __init__(self, input_shape, output_dim):
        # Define the Leaky ReLU activation function


        # Define the CNN model
              # Créez un modèle séquentiel
            self.input_shape = input_shape
            model = keras.Sequential()

            # Convolutional layers
            model.add(layers.Conv2D(32, (2, 3), activation=layers.LeakyReLU(alpha=0.1), input_shape=input_shape, padding='same'))


            model.add(layers.Conv2D(64, (2, 3), activation=layers.LeakyReLU(alpha=0.1), padding='same'))


            model.add(layers.Conv2D(128, (2, 3), activation=layers.LeakyReLU(alpha=0.1), padding='same'))

            #Max pooling 2d
            #model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            # Flatten layer to transition from convolutional layers to fully connected layers
            model.add(layers.Flatten())

            # Fully connected layers
            model.add(layers.Dense(1500, activation=layers.LeakyReLU(alpha=0.1)))
            model.add(layers.Dense(1500, activation=layers.LeakyReLU(alpha=0.1)))
            model.add(layers.Dense(1500, activation=layers.LeakyReLU(alpha=0.1)))
            model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
            model.add(layers.Dense(output_dim, activation='sigmoid'))  # Binary classification (sigmoid activation)

                    # Compiler le modèle avec une fonction de perte (loss) appropriée et un optimiseur
            model.compile(
                optimizer='adam', loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.4),tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.45)],run_eagerly=True
            )

            # Résumé du modèle
            model.summary()
            self.model = model

        def train(
            self, X_train, y_train, epochs=30, batch_size=1, validation_data=None
        ):
            # Entraînez le modèle avec vos données d'entraînement et vos étiquettes
            self.X_train = X_train
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
            )
            # Affichez un résumé du modèle
            self.model.summary()
            return history

        def saveModel(self, name):
            self.model.save("./saved/" + name)

    def evaluate(self, X_test, y_test_true, threshold=0.5):
        tf.config.run_functions_eagerly(True)

        y_predicted = self.model.predict(X_test)


        randomTestIndex = random.randint(0, len(y_predicted))
        PredictedStepPlot().evaluate(
            y_test_true[randomTestIndex], y_predicted[randomTestIndex], threshold = threshold
        )

        y_test_pred_binary = (y_predicted > threshold).astype(int)
        print("LES METRIQUES SUIVANTES SONT CALCULEES APRES  AVOIR TRANSFORMER LES DONNEES DE SORTIES EN SORTIES BINAIRE AVEC UN SEUIL DE "+str(threshold))
        MSEEvaluateur().evaluate(y_test_true, y_test_pred_binary)
        RMSEEvaluateur().evaluate(y_test_true, y_test_pred_binary)
        R2Score().evaluate(y_test_true, y_test_pred_binary)
        Accuracy().evaluate(y_test_true, y_test_pred_binary)

    def evaluate_from_simulation(self, number_samples):
        # Utiliser radarsimpy simulator
        pass
    def predict(self, test_data):
        return self.model.predict(test_data)

    def load(self, name, custom_loss=False):
        if custom_loss==False:
            self.model = keras.models.load_model(os.getcwd() + "/Models/saved/" + name)
        else:
            self.model = keras.models.load_model(os.getcwd() + "/Models/saved/" + name, custom_objects={custom_loss: self.Trainer.custom_loss})

if __name__ == "__main__":
    absolutePath = "C:/Users/Younes srh/Desktop/I3/ProjetRadarIA/Data/"
    data_loader = DataLoader([absolutePath +"Dataset_X7979_3-2S.csv",absolutePath + "Dataset_X8004_2S.csv", absolutePath + "Dataset_X2922_2S.csv",absolutePath + "Dataset_X9193_3-2S.csv", absolutePath + "Dataset_X4559_3-2S.csv",absolutePath +"Dataset_X2599_3-2S.csv",absolutePath+"Dataset_X_3S_30-30.csv",absolutePath+"Dataset_X4523_3S.csv"],
                             [absolutePath +"Dataset_y7979_3-2S.csv",absolutePath + "Dataset_y8004_2S.csv", absolutePath + "Dataset_y2922_2S.csv",absolutePath + "Dataset_y9193_3-2S.csv", absolutePath + "Dataset_y4559_3-2S.csv",absolutePath +"Dataset_y2599_3-2S.csv",absolutePath+"Dataset_y_3S_30-30.csv",absolutePath+"Dataset_y4523_3S.csv"])
    data, labels = data_loader.load_data()
    #radar_dataset = RealImaginaryXDataSet(data, labels, 0.4, appended_snr=True)
    radar_dataset = RealImaginaryRxxDataSet(data, labels, 0.4, appended_snr=True)
    trainer = DocuCNNModel.Trainer((10,10,2), 181)
    print("shape de la dataset d'entrainement :")
    print(radar_dataset.X_train.shape)
    history = trainer.train(
        radar_dataset.X_train,
        radar_dataset.y_train,
        epochs=50,
        batch_size=1000,
        validation_data=(radar_dataset.X_validation, radar_dataset.y_validation),

    )
    learningCurvePloter = LearningCurvesPlot(metrics = ["binary_io_u"])
    learningCurvePloter.evaluate(history)
    trainer.saveModel("CNN_final_docu5_RealImaginaryRxx")



