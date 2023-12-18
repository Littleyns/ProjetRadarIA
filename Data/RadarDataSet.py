from sklearn.model_selection import train_test_split
import numpy as np

from Models.BasicAutoEncoder import BasicAutoEncoder
from PreProcessing.domaines.passage_freq import get_signal_frequentiel
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PreProcessing.utils import augmentDataInterp


class RadarDataSet:
    def __init__(self, data, labels, test_size, scaler=StandardScaler(), appended_snr=False):
        self.scaler = scaler
        self.X = data
        self.y = labels
        self.test_size = test_size

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        (
            self.X_validation,
            self.X_test,
            self.y_validation,
            self.y_test,
        ) = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=42)

        #Séparation du SNR
        if appended_snr:
            self.snr_y_test = self.y_test[:,-1]
            self.snr_y_train =self.y_train[:,-1]
            self.snr_y_validation = self.y_validation[:,-1]
            self.y_test = self.y_test[:,:-1]
            self.y_train =self.y_train[:,:-1]
            self.y_validation = self.y_validation[:,:-1]

        # mise à l'echelle
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.fit_transform(self.X_test)
        self.X_validation = scaler.fit_transform(self.X_validation)

        self.X_freq_train = []
        self.X_freq_test = []
        self.X_freq = []

        self.X_real_train = self.X_train[:, :100]  # partie reelle
        self.X_im_train = self.X_train[:, 100:]  # partie imaginaire

        self.y_train_360 = augmentDataInterp(self.y_train, 360)
        self.y_test_360 = augmentDataInterp(self.y_test, 360)

    def add_frequential_data(self):
        for signal in self.X:
            self.X_freq += (get_signal_frequentiel(signal),)
        self.X_freq = np.array(self.X_freq)
        self.X_freq_train, self.X_freq_test, empty1, empty2 = train_test_split(
            self.X_freq, self.y, test_size=self.test_size, random_state=42
        )

    def load_Rxx(self):
        self.Rxx_train = recreate_rxx_abs(self.X_train)
        self.Rxx_test = recreate_rxx_abs(self.X_test)
        self.Rxx_validation = recreate_rxx_abs(self.X_validation)


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

def recreate_rxx_abs(data):
    real_part = data[:, : data.shape[1] // 2].reshape(-1, 10, 10)
    imag_part = data[:, data.shape[1] // 2:].reshape(-1, 10, 10)
    return np.abs(real_part + 1j * imag_part)

def merged_data_format(data):
    # data1=reel data2=img
    # alt
    # result = np.vstack((row1, row2) for row1, row2 in zip(data1, data2))

    # matrice2
    # merged_data = np.array([np.concatenate((row1, row2)) for row1, row2 in zip(data1, data2)])
    # result1 = merged_data.reshape(data1.shape[0], -1, data1.shape[1])
    pass;