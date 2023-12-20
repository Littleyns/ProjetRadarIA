import random

import tensorflow as tf
from keras import Sequential, layers
import keras
import os

from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet
from Evaluation.plots import PredictedStepPlot, LearningCurvesPlot
from Evaluation.statistic_errors import MSEEvaluateur, RMSEEvaluateur, R2Score, Accuracy
from sklearn.preprocessing import StandardScaler
from Models.BasicAutoEncoder import BasicAutoEncoder


class DocuCNNModel:
    def __init__(self, model=None):
        self.model = model

    class Trainer:
        def leaky_relu(x, alpha=0.01):
            return tf.maximum(alpha * x, x)
        def __init__(self, input_shape, output_dim):
        # Define the Leaky ReLU activation function


        # Define the CNN model
              # Créez un modèle séquentiel
            self.input_shape = input_shape
            model = keras.Sequential()

            # Convolutional layers
            model.add(layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.1), input_shape=(height, width, channels), padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.1), padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.1), padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))

            # Flatten layer to transition from convolutional layers to fully connected layers
            model.add(layers.Flatten())

            # Fully connected layers
            model.add(layers.Dense(1500, activation=layers.LeakyReLU(alpha=0.1)))
            model.add(layers.Dense(1500, activation=layers.LeakyReLU(alpha=0.1)))
            model.add(layers.Dense(1500, activation=layers.LeakyReLU(alpha=0.1)))
            model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
            model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (sigmoid activation)

                    # Compiler le modèle avec une fonction de perte (loss) appropriée et un optimiseur
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
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
        print("LES METRIQUES SUIVANTES SONT CALCULEES APRES AVOIR TRANSFORMER LES DONNEES DE SORTIES EN SORTIES BINAIRE AVEC UN SEUIL DE "+str(threshold))
        MSEEvaluateur().evaluate(y_test_true, y_test_pred_binary)
        RMSEEvaluateur().evaluate(y_test_true, y_test_pred_binary)
        R2Score().evaluate(y_test_true, y_test_pred_binary)
        Accuracy().evaluate(y_test_true, y_test_pred_binary)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def load(self, name):
        self.model = keras.models.load_model(os.getcwd() + "/Models/saved/" + name)


if __name__ == "__main__":
    data_loader = DataLoader("C:/Users/HP/Desktop/rad/Data/Dataset_X6687.csv","C:/Users/HP/Desktop/rad/Data/Dataset_y6687.csv")
    data, labels = data_loader.load_data()
    radar_dataset = RadarDataSet(data, labels, 0.4)

    trainer = DocuCNNModel.Trainer(radar_dataset.X_train.shape, 181)
    history = trainer.train(
        radar_dataset.X_train,
        radar_dataset.y_train,
        epochs=30,
        batch_size=50,
        validation_data=(radar_dataset.X_validation, radar_dataset.y_validation),

    )
    learningCurvePloter = LearningCurvesPlot()
    learningCurvePloter.evaluate(history)
    trainer.saveModel("CNN15_bcross_b50_e30_sigmoid_adam")



