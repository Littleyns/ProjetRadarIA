import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, log_loss, accuracy_score,roc_auc_score
import pandas as pd
from Evaluation.Evaluateur import Evaluateur


class PredictedStepPlot(Evaluateur):
    def evaluate(self, y_true, y_pred, threshold) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Resultat attendu vs resultat prédit')
        ax1.set_title("attendu")
        ax1.step(np.arange(0, y_pred.shape[0], 1), y_true)
        ax2.set_title("Predit")
        ax2.step(np.arange(0, y_pred.shape[0], 1), y_pred, color="darkorange")
        ax2.axhline(y=threshold, color='red', linestyle='--', label='seuil binaire')
        plt.figure()
        plt.title("Cote à cote")
        plt.step(np.arange(0, y_pred.shape[0], 1),y_true)
        plt.step(np.arange(0, y_pred.shape[0], 1), y_pred)

class LearningCurvesPlot(Evaluateur):
    def evaluate(self, history):
        plt.figure(figsize=(12, 4))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.show()

class ErrorOfSNRPlot(Evaluateur):
    def evaluate(self, snr_values, y_true, y_pred, errorFunc = 'f1_score'):

        df = pd.DataFrame({'SNR': snr_values, 'y_true': list(y_true), 'y_pred': list(y_pred)})

        # Calculez le RMSE pour chaque groupe de SNR
        if(errorFunc == 'accuracy_score'):
            rmse_values = df.groupby('SNR').apply(lambda group: np.sqrt(
                accuracy_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()))))
        elif errorFunc == 'roc_auc_score':
            rmse_values = df.groupby('SNR').apply(lambda group: np.sqrt(
                roc_auc_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()),average='micro')))
        else:
            rmse_values = df.groupby('SNR').apply(lambda group: np.sqrt(
                f1_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()),average='micro')))

        # Créez le diagramme à barres
        plt.plot(rmse_values.index, rmse_values, color='blue')

        # Ajoutez des étiquettes et un titre
        plt.xlabel('SNR')
        plt.ylabel(errorFunc)
        plt.title('score/error par SNR')

        # Affichez le diagramme
        plt.show()
