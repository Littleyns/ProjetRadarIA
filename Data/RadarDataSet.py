from sklearn.model_selection import train_test_split
import numpy as np
from PreProcessing.domaines.passage_freq import get_signal_frequentiel
import matplotlib.pyplot as plt
class RadarDataSet:
    def __init__(self, data, labels, test_size):
        self.X = data
        self.y = labels
        self.test_size = test_size;
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
        self.X_freq_train = []
        self.X_freq_test = []
        self.X_freq = []

    def add_frequential_data(self):
        for signal in self.X:
            self.X_freq += get_signal_frequentiel(signal),
        self.X_freq = np.array(self.X_freq)
        self.X_freq_train, self.X_freq_test, empty1, empty2 = train_test_split(self.X_freq, self.y, test_size=self.test_size, random_state=42)


    def plot(self, index ):
        plt.figure(figsize=(10, 6))
        plt.plot(self.X[index], label='Signal')  # Tracez le signal
        # Ajoutez des titres et des légendes
        plt.title('Tracé du signal')
        plt.xlabel('Temps')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.figure(figsize=(10, 6))
        plt.step(np.arange(0, self.y[index].shape[0], 1), self.y[index])
        plt.title('Label du signal (position objet)')
        plt.xlabel('Angle')

        if (len(self.X_freq) != 0 and type(self.X_freq[index])!=type(None)):
            plt.figure(figsize=(10, 6))
            frequence = np.fft.fftfreq(self.X_freq[index].shape[0])
            plt.plot(frequence, np.abs(self.X_freq[index]))
            plt.title("Spectre de fréquence du signal radar")
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude")
            plt.grid(True)

        # Affichez le graphique
        plt.show()
