import keras
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.constants import c
from Models.DocuCNNModel import DocuCNNModel
from simulateur_prof.sim_utils import get_mode_calcul
from simulateur_prof.Main_NSources import lfm_waveform, radar_get_matched_filter,receive_array
import random
from sklearn.preprocessing import StandardScaler
from simulateur_prof.MatchedFilter import matched_filter
from simulateur_prof.RadarChannel import radar_channel
from simulateur_prof.awgNoise import awgn_noise
from Models.RNNmodel import RNNModel

if get_mode_calcul() == "gpu":
    import cupy as np

else:
    import numpy as np
import numpy as nump
class VisualSimu:
    def random_source1_inside_range(self):
        doa = random.randint(-30,31)
        R = random.randint(self.min_sources_distance_to_radar,self.max_sources_distance_to_radar)

        speed_polar = [(1,-1)[nump.random.randint(0,2)==1],(1,-1)[nump.random.randint(0,2)==1]]
        if abs(int((self.max_sources_distance_to_radar-R)/self.min_simulation_duration)) > abs(int((self.min_sources_distance_to_radar-R)/self.min_simulation_duration)):
            speed_polar[0] = round(random.uniform(0, int((self.max_sources_distance_to_radar-R)/self.min_simulation_duration)),1)
        else:
            speed_polar[0] = -round(random.uniform(0, abs(int((self.min_sources_distance_to_radar-R)/self.min_simulation_duration))),1)


        speed_polar[1] *= random.randint(1,int(abs((speed_polar[1]*60)-doa) / self.min_simulation_duration))
        pos_polar = [R,doa]
        return pos_polar, speed_polar
    def __init__(self):
        # Paramètres de la simulation

        self.wave_speed = 0.05  # Vitesse de propagation de l'onde
        self.min_simulation_duration = 10
        self.min_sources_distance_to_radar= 5
        self.max_sources_distance_to_radar = 30

        self.SNR = 30
        self.fc = 2.725e9  # fréquence centrale (Hz)
        self.BW = 10e6  # largeur de bande (Hz)
        self.Te = 1.4e-3  # Durée de modulation (s) (Pulse width)
        self.T = 4.1e-3  # Période de répétition des impulsions (s)
        self.fs = 4 * self.BW  # % Fréquence d'échantillonnage (Hz)
        self.lambda_ = c / self.fc
        self.N = 10  # Nombre de radars dans l'array
        self.d = 0.5  # Espacement entre les radars
        self.LFMDir = "Up"
        self.LFM, self.temps = lfm_waveform(self.BW, self.Te, self.fs, self.LFMDir, self.T)
        self.txSig = self.LFM.reshape(1, self.LFM.shape[0])
        # Radar Matched Filter
        self.mf_coeff, self.H, self.winTemp = radar_get_matched_filter(self.LFM)
        self.M = 2 #nombre de sources
        self.sources = [] #pour chaque source [tupple_pos, tupple_vitesse]
        self.sources_instant_positions = []
        self.history_positions = []

        Rfix = [10,17,26] #Fixation des rayons R

        for i in range(self.M):
            self.sources+=self.random_source1_inside_range(),
            self.sources[i][0][0] = Rfix[i]
            self.sources_instant_positions += self.sources[i][0],

        self.model = DocuCNNModel()
        keras.utils.get_custom_objects()['angleSensitiveCustomLoss'] = self.model.Trainer.angleSensitiveCustomLoss
        self.model.load("CNN_docu10_XRI_e30_b350_anglesensitive_2S", custom_loss="angleSensitiveCustomLoss")

        self.speedModel = RNNModel()
        self.speedModel.load("RNNModel1")


    def predict_doa(self, Rxx):
        scaler = StandardScaler()
        Rxx = nump.concatenate([Rxx.real.flatten(), Rxx.imag.flatten()]).reshape(
            1, -1
        )
        data_im = Rxx[:, 100:]
        data_real = Rxx[:, :100]
        Rxx = nump.stack((data_real,data_im), axis=1)
        Rxx = scaler.fit_transform(Rxx.squeeze())
        if get_mode_calcul() == "gpu":
            doa = self.model.predict(np.array([Rxx]).get())
        else:
            doa = self.model.predict(np.array([Rxx]))
        return doa

    # Fonction pour calculer l'intensité de l'onde reçue par chaque radar
    def calculate_wave_intensity(self, radar_positions, source_positions, time):
        wave_intensity = np.zeros_like(radar_positions, dtype=float)
        for source_position in source_positions:
            distance = radar_positions - source_position
            time_delay = distance / self.wave_speed
            wave_intensity += np.sin(2 * np.pi * (time - time_delay))
        return wave_intensity

    def get_signal_rxx(self):
        R = np.array( [    [source[0]] for source in self.sources_instant_positions     ] )
        doa = np.array([source[1] for source in self.sources_instant_positions])
        txSig_chan = radar_channel(self.txSig, self.fc, self.fs, R, two_way=True)
        sigPower = np.sum(np.abs(txSig_chan) ** 2, axis=1) / txSig_chan.shape[1]
        sigPower = np.sqrt(sigPower[0] / sigPower[1:])[0]
        txSig_chan[1:, :] *= sigPower
        sig_rx = receive_array(txSig_chan, self.N, self.d, doa)[0]
        sigNoise = awgn_noise(sig_rx, self.SNR)

        # Compression
        X = matched_filter(sigNoise, self.H)  # replace sig_rx by sigNoise
        # Matrice de corrélation
        Rxx = (1 / X.shape[1]) * np.dot(X, X.conj().T)
        if get_mode_calcul() == "gpu":
            Rxx = np.array([Rxx]).get()
        else:
            Rxx = np.array([Rxx])
        return Rxx

    def update(self, frame, source1_position=None):

        Rxx = self.get_signal_rxx()
        #y_music_predicted_angles, y_music_predicted_multilabel = MUSICModel(self.M, nump.arange(-90, 90, 0.1)).predict(Rxx)
        doa = self.predict_doa(Rxx)
        doa = np.sort(np.argsort(doa, axis=1)[:, -self.M:]-90,axis=1).squeeze()
        self.ax.clear()
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_thetamin(-90)
        self.ax.set_thetamax(90)

        strPrintTrue = ''
        strPrintPred = ''
        self.history_positions+= [],
        for i in range(self.M):
            # Pas besoin de faire le mouvement sur l'axe y
            #self.sources_instant_positions[i][0]= round(self.sources_instant_positions[i][0]+self.sources[i][1][0],1) #update R

            self.sources_instant_positions[i][1] += self.sources[i][1][1] #Update theta
            self.history_positions[-1]+=self.sources_instant_positions[i].copy(),
            # Ajouter les points représentant les positions des sources
            self.ax.scatter(nump.radians(self.sources_instant_positions[i][1]), self.sources_instant_positions[i][0], c='r',marker=(i+1)*4,s=100)  # 'ro' représente le rouge
            self.ax.annotate(f'S{i}', xy=(nump.radians(self.sources_instant_positions[i][1]), self.sources_instant_positions[i][0]), xytext=(nump.radians(self.sources_instant_positions[i][1]), self.sources_instant_positions[i][0]-0.5))
            strPrintTrue += f"S{i}: pos"+str(self.sources_instant_positions[i])+" V="+str(self.sources[i][1])+"\n"
            strPrintPred += f"pos" + f"(?, {str(doa[i])})"
        radar_positions = np.linspace(-60, 60, self.N)
        #wave_intensity = self.calculate_wave_intensity(radar_positions, [self.source1_position, self.source2_position], frame)
        self.ax.text(0,1,"True positions and speed \n"+strPrintTrue,fontsize="medium", fontweight="bold", bbox= dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),transform=self.ax.transAxes, horizontalalignment='left')

        if len(self.history_positions)>=5:
            timeSerie = np.empty((0, 181))
            series = []
            for t in range(len(self.history_positions)-5, len(self.history_positions)):
                theta = np.zeros(181)
                theta[[int(self.history_positions[t][i][1]) for i in range(len(self.history_positions[t]))]] = 1
                series.append(theta)
            timeSerie = np.vstack([timeSerie] + series)
            predicted_speed = np.round(self.speedModel.predict(np.array([timeSerie]))).squeeze()
            for i in range(len(self.history_positions[-5])):
                strPrintPred += f" V(S{i}=" + str(predicted_speed[int(self.history_positions[-5][i][1]+90)]) + "\n"
        self.ax.text(0,0,"Predicted positions and speed\n"+strPrintPred,fontsize="medium", fontweight="bold", bbox= dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),transform=self.ax.transAxes, horizontalalignment='left')

        # Tracer l'intensité de l'onde
        #self.ax.plot(np.radians(np.linspace(-60, 60, self.num_radars)), wave_intensity, color='blue')

    def start(self):
        # Création de la figure et de l'axe polar
        fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Animation
        ani = FuncAnimation(fig, self.update, frames=np.arange(0, 100), interval=10000)

        plt.show()

if __name__ == "__main__":
    VisualSimu().start()