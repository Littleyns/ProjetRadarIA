import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from PreProcessing.domaines.passage_freq import get_signal_frequentiel
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PreProcessing.utils import augmentDataInterp, data_to_complex


class RadarDataSet:
    def __init__(self, data, labels, test_size, scaler=StandardScaler(), appended_snr=False):
        self.scaler = scaler
        self.n = data.shape[0]
        self.X = data
        self.y = labels
        self.test_size = test_size
        self.seedDecoupageData = 42


        if scaler!=None:
            # mise à l'echelle
            self.X = scaler.fit_transform(self.X)

        self.X_train, self.X_test,self.X_validation, self.y_train, self.y_test, self.y_validation = self.data_split(self.X, self.y)


        #Séparation du SNR
        if appended_snr:
            self.snr_y = self.y[:, -1]
            self.snr_y_test = self.y_test[:,-1]
            self.snr_y_train =self.y_train[:,-1]
            self.snr_y_validation = self.y_validation[:,-1]
            self.y = self.y[:, :-1]
            self.y_test = self.y_test[:,:-1]
            self.y_train =self.y_train[:,:-1]
            self.y_validation = self.y_validation[:,:-1]


        #self.y_train_360 = augmentDataInterp(self.y_train, 360)
        #self.y_test_360 = augmentDataInterp(self.y_test, 360)

    def data_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seedDecoupageData
        )
        (
            X_validation,
            X_test,
            y_validation,
            y_test,
        ) = train_test_split(X_test, y_test, test_size=0.5, random_state=self.seedDecoupageData)
        return X_train, X_test, X_validation, y_train, y_test, y_validation

    def load_Rxx(self):
        self.X = recreate_rxx(self.X)
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = self.data_split(self.X, self.y)

    def load_Rxx_r_i(self): #Charge une dataset sous forme de Rxx reconstruit(10,10,2)
        real_part = self.X[:, : self.X.shape[1] // 2].reshape(-1, 10, 10)
        imag_part = self.X[:, self.X.shape[1] // 2:].reshape(-1, 10, 10)
        Rxx_im_complex = np.stack((real_part, abs(imag_part)), axis=1)  # (n,2,10,10)
        Rxx_im_complex = np.transpose(Rxx_im_complex, (0, 2, 3, 1))  # (n,10,10,2)
        self.X = Rxx_im_complex
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = self.data_split(Rxx_im_complex, self.y)

    def load_X_r_i(self): #Charge une dataset sous la forme (2,100) real,imaginaire
        self.X = data_to_complex(self.X)
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = self.data_split(self.X, self.y)
    def plot(self, index):
        plt.figure(figsize=(10, 6))
        plt.plot(self.X[index], label="Signal")  # Tracez le signal
        # Ajoutez des titres et des légendes
        plt.title("Tracé du signal")
        plt.xlabel("Temps")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.figure(figsize=(10, 6))
        plt.step(np.arange(0, self.y[index].shape[0], 1), self.y[index])
        plt.title("Label du signal (position objet)")
        plt.xlabel("Angle")

        if len(self.X_freq) != 0 and type(self.X_freq[index]) != type(None):
            plt.figure(figsize=(10, 6))
            frequence = np.fft.fftfreq(self.X_freq[index].shape[0])
            plt.plot(frequence, np.abs(self.X_freq[index]))
            plt.title("Spectre de fréquence du signal radar")
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude")
            plt.grid(True)

        # Affichez le graphique
        plt.show()

    def save_by_snr(self, f_name):
        df = pd.DataFrame({'SNR': self.snr_values, 'y': list(self.y), 'X': list(self.X)})
        grouped_data = df.groupby('SNR')

        # Parcourir chaque groupe et enregistrer dans un fichier CSV
        for group_name, group_data in grouped_data:
            group_data.to_csv(f'../Data/{f_name}/SNR_{group_name}.csv', index=False)

def recreate_rxx_abs(data):
    real_part = data[:, : data.shape[1] // 2].reshape(-1, 10, 10)
    imag_part = data[:, data.shape[1] // 2:].reshape(-1, 10, 10)
    return np.abs(real_part + 1j * imag_part)

def recreate_rxx(data):
    real_part = data[:, : data.shape[1] // 2].reshape(-1, 10, 10)
    imag_part = data[:, data.shape[1] // 2:].reshape(-1, 10, 10)
    return real_part + 1j * imag_part

def merged_data_format(data):
    # data1=reel data2=img
    # alt
    # result = np.vstack((row1, row2) for row1, row2 in zip(data1, data2))

    # matrice2
    # merged_data = np.array([np.concatenate((row1, row2)) for row1, row2 in zip(data1, data2)])
    # result1 = merged_data.reshape(data1.shape[0], -1, data1.shape[1])
    pass;


class AlternatedRealImaginaryDataSet(RadarDataSet): #DataSet sous la forme [R1,Im1,R2,Im2,...] shape => (n,200)
    def __init__(self, data, labels, test_size, scaler=StandardScaler(), appended_snr=False):
        super().__init__(data, labels, test_size, scaler, appended_snr)
        real_part = self.X[:, :100]
        imag_part = self.X[:, 100:]
        self.X = np.reshape(np.stack((real_part, imag_part), axis=-1), (self.n, 200))

class RealImaginaryRxxDataSet(RadarDataSet):  #Forme initiale de la matrice de corrélation sur deux dimensions pour la partie reele et imaginaire shape => (n,10,10,2)
    def __init__(self, data, labels, test_size, scaler=StandardScaler(), appended_snr=False):
        super().__init__(data, labels, test_size, scaler, appended_snr)
        real_part = self.X[:, : self.X.shape[1] // 2].reshape(-1, 10, 10)
        imag_part = self.X[:, self.X.shape[1] // 2:].reshape(-1, 10, 10)
        Rxx_im_complex = np.stack((real_part, abs(imag_part)), axis=1)  # (n,2,10,10)
        Rxx_im_complex = np.transpose(Rxx_im_complex, (0, 2, 3, 1))  # (n,10,10,2)
        self.X = Rxx_im_complex
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = self.data_split(
            Rxx_im_complex, self.y)

class RxxDataSet(RadarDataSet):#Forme initiale de la matrice de corrélation complexe shape=> (n,10,10)
    def __init__(self, data, labels, test_size, scaler=StandardScaler(), appended_snr=False):
        super().__init__(data, labels, test_size, scaler, appended_snr)
        self.X = recreate_rxx(self.X)
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = self.data_split(
            self.X, self.y)

class RealImaginaryXDataSet(RadarDataSet):# Dataset de base sur deux dimensions pour la partie reele et imaginaire shape=> (n,2,100)
    def __init__(self, data, labels, test_size, scaler=StandardScaler(), appended_snr=False):
        super().__init__(data, labels, test_size, scaler, appended_snr)
        self.X = data_to_complex(self.X)
        self.X_train, self.X_test, self.X_validation, self.y_train, self.y_test, self.y_validation = self.data_split(
            self.X, self.y)