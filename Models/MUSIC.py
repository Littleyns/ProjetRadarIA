import numpy as np
import matplotlib.pyplot as plt

from Data.DataLoader import DataLoader
from Data.RadarDataSet import RadarDataSet, recreate_rxx, RealImaginaryRxxDataSet, RxxDataSet
from Evaluation.statistic_errors import RMSEEvaluateur


class MUSICModel:
    def music_doa(self, Rxx, theta, M):
        """
        Calcule les angles d'arrivées des signaux RF non-corrélées en présence de bruit additif gaussien.

        Parameters:
        - Rxx: Matrice d'auto-corrélation du réseau entier
        - theta: Liste d'angles d'arrivées des signaux
        - M: Nombre de signaux

        Returns:
        - Pmusic: Pseudo-spectre MUSIC
        - EN: Sous-espace bruit
        """

        # Décomposition en vecteurs singuliers

        Dia, V = np.linalg.eig(Rxx)
        # Trie des valeurs propres par ordre croissant
        Index = np.argsort(Dia)

        # Calcul de la matrice des (N-M) vecteurs propres associés au sous-espace bruit
        EN = V[:, Index[: len(Rxx) - M]]

        # Calcul du pseudo-spectre MUSIC
        Pmusic = np.zeros(len(theta))
        n = np.arange(1, len(Rxx) + 1).reshape(10, 1)

        for k in range(len(theta)):
            a = np.exp(1j * (n - 1) * np.pi * np.sin(np.radians(theta[k])))
            Pmusic[k] = 1 / np.abs(
                np.dot(a.conj().transpose(), EN) @ np.dot(EN.conj().transpose(), a)
            )  # Pseudo-spectre MUSIC

        return Pmusic, EN

    def __init__(self, N_sources, theta):
        self.N_sources = N_sources
        self.theta = theta

    def predict(self,Rxx):
        y_music_predicted_angles = [] # format [angle1, angle2] (,N_source)
        y_music_predicted_multilabel = [] #format [0,0,0,1,....,1] (,180)
        for i in range(Rxx.shape[0]):
            Pmusic, EN = self.music_doa(Rxx[i], self.theta, self.N_sources)
            Pmusiclog10 = np.log10(Pmusic)
            musicPredictedAngles = (np.argsort(Pmusiclog10)[::-1][:2]*round(self.theta[1] - self.theta[0],2))-90
            y_music_predicted_angles += musicPredictedAngles,
            thetaRes = np.zeros(181)
            thetaRes[[int(doa)+90 for doa in musicPredictedAngles]] = 1
            y_music_predicted_multilabel += thetaRes,
        return np.asarray(y_music_predicted_angles), np.asarray(y_music_predicted_multilabel)

    def plot(self, Pmusiclog10):
        plt.figure()

        plt.plot(self.theta, 10 * Pmusiclog10)
        plt.grid(True)
        plt.title("DOAs cibles")
        plt.xlabel("DOA (degree)")
        plt.ylabel("|P(theta)|")
        plt.show()


if __name__ == "__main__":
    data_loader = DataLoader("../Data/Dataset_X2922_2S.csv", "../Data/Dataset_y2922_2S.csv")
    data, labels = data_loader.load_data()
    radar_dataset = RxxDataSet(data, labels, 0.1,appended_snr=True, scaler=None)  # 0.2 is the test size ( 80% train data, 20% test data)
    musicModel = MUSICModel(2, np.arange(-90, 90, 0.1))
    y_music_predicted_angles, y_music_predicted_multilabel = musicModel.predict(radar_dataset.X_test)
    test = RMSEEvaluateur().evaluate(radar_dataset.y_test, y_music_predicted_multilabel, 2)
    print(test)