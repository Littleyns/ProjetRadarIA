import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, log_loss, accuracy_score,roc_auc_score
import pandas as pd
from Evaluation.Evaluateur import Evaluateur
from Evaluation.statistic_errors import RMSEEvaluateur


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
    def __init__(self, metrics):
        self.metrics = metrics
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
        for metric in self.metrics:
            plt.plot(history.history[metric], label='Training '+metric)
            plt.plot(history.history['val_'+metric], label='Validation '+metric)
        plt.legend()
        plt.title('Metrics vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')

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

class MusicVsModelBySNR(Evaluateur):

    def __init__(self, errorFunc = 'f1_score', Nsources=2, step=1):
        self.step = step
        self.errorFunc = errorFunc
        self.Nsources = Nsources
    def evaluate(self, snr_values_music, y_true_music, y_pred_music, snr_values_model, y_true_model, y_pred_model):


        #Pmusic, EN = music_doa(Rxx, thetaM, Nsources)
        df_music= pd.DataFrame({'SNR': snr_values_music, 'y_true': list(y_true_music), 'y_pred': list(y_pred_music)})
        df = pd.DataFrame({'SNR': snr_values_model, 'y_true': list(y_true_model), 'y_pred': list(y_pred_model)})
        if self.step != 1:
            df = df.query(f"SNR %{self.step} == 0")
            df_music = df_music.query(f"SNR %{self.step} == 0")
        # Calculez le RMSE pour chaque groupe de SNR
        if(self.errorFunc == 'accuracy_score'):
            rmse_values_model = df.groupby('SNR').apply(lambda group: np.sqrt(
                accuracy_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()))))
            rmse_values_music=df_music.groupby('SNR').apply(lambda group: np.sqrt(
                accuracy_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()))))
        elif self.errorFunc == 'roc_auc_score':
            rmse_values_model = df.groupby('SNR').apply(lambda group: np.sqrt(
                roc_auc_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()),average='micro')))
            rmse_values_music=df_music.groupby('SNR').apply(lambda group: np.sqrt(
                roc_auc_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()))))
        elif self.errorFunc == 'rmse':
            rmse_values_model = df.groupby('SNR').apply(lambda group: np.sqrt(RMSEEvaluateur().evaluate(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()),Nsources=self.Nsources, verbose=False)))
            rmse_values_music = df_music.groupby('SNR').apply(lambda group: np.sqrt(RMSEEvaluateur().evaluate(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()),Nsources=self.Nsources, verbose=False)))
        else:
            rmse_values_model = df.groupby('SNR').apply(lambda group: np.sqrt(
                f1_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()),average='micro')))
            rmse_values_music=df_music.groupby('SNR').apply(lambda group: np.sqrt(
                f1_score(np.array(group['y_true'].tolist()), np.array(group['y_pred'].tolist()))))

        # Créez le diagramme à barres
        plt.title("Music VS CNN Model "+str(self.Nsources)+" Sources")
        plt.plot(rmse_values_model.index, rmse_values_model, color='blue', label="CNN Model")
        plt.plot(rmse_values_music.index, rmse_values_music, color='darkorange', label = "MUSIC")
        plt.legend()
        # Ajoutez des étiquettes et un titre
        plt.xlabel('SNR')
        plt.ylabel(self.errorFunc)
        plt.title('score/error par SNR')

        # Affichez le diagramme
        plt.show()
