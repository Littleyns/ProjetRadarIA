import matplotlib.pyplot as plt
import numpy as np
class RadarData:
    def __init__(self, signal, label):
        self.signal_domaine_frequentiel = None
        self.signal_domaine_temporel = signal # Utiliser un preprocessor fourier pour le charger
        self.label = label

    def setSignalDomaineFrequentiel(self, signal_domaine_frequentiel):
        self.signal_domaine_frequentiel = signal_domaine_frequentiel
    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.signal_domaine_temporel, label='Signal')  # Tracez le signal
        # Ajoutez des titres et des légendes
        plt.title('Tracé du signal')
        plt.xlabel('Temps')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.figure(figsize=(10, 6))
        plt.step(np.arange(0, self.label.shape[0], 1), self.label)
        plt.title('Label du signal (position objet)')
        plt.xlabel('Angle')

        if (type(self.signal_domaine_frequentiel) != type(None)):
            plt.figure(figsize=(10, 6))
            frequence = np.fft.fftfreq(self.signal_domaine_temporel.shape[0])
            plt.plot(frequence, np.abs(self.signal_domaine_frequentiel))
            plt.title("Spectre de fréquence du signal radar")
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Amplitude")
            plt.grid(True)

        # Affichez le graphique
        plt.show()
    # Ajoutez d'autres méthodes de gestion des données radar