import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.signal import windows
from scipy.signal import chirp, correlate
from MatchedFilter import matched_filter
from Music_doa import music_doa
from RadarChannel import radar_channel
from ULA_array import ula_array
from awgNoise import awgn_noise
from os import environ
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
import random
from sim_utils import get_mode_calcul
import chainer
chainer.print_runtime_info()
# Enregistrez le temps de début
start_time = time.time()
executor = ThreadPoolExecutor()
# report the number of worker threads chosen by default
print(executor._max_workers)
environ["OMP_NUM_THREADS"] = str(executor._max_workers)
MODE_CALCUL = get_mode_calcul()
print("MODE_CALCUL: "+MODE_CALCUL)
if MODE_CALCUL == "gpu":
    import cupy as np
    import numpy as nump
else:
    import numpy as np

def receive_array(x, N, d, theta):
    A = ula_array(N, d, theta)
    Na = A.shape[1]
    lx, cx = x.shape

    if Na == lx:
        y = np.dot(A, x)
    elif Na == cx:
        y = np.dot(A, x.T)
    else:
        raise ValueError("Error.")

    return y, x

def lfm_waveform(BW, Te, fs, chirp_dir, T, nbre_tirs=1):
    f0 = 0
    Ts = 1 / fs
    npoints = int(Te / Ts)
    K = BW / Te
    A = 1

    if chirp_dir == "Symmetry":
        t = np.linspace(-0.5 * Te, 0.5 * Te, npoints)
    elif chirp_dir == "Up":
        t = np.linspace(0, Te, npoints)
    elif chirp_dir == "Down":
        t = np.linspace(-Te, 0, npoints)

    lfm_sig = A * np.exp(1j * 2 * np.pi * (f0 * t + (K / 2) * t**2))
    temps = np.linspace(0, Te, npoints)

    Npointsrx = int((T - Te) / Ts)
    signal = np.concatenate([lfm_sig, np.zeros(Npointsrx - 1)])

    lfm_signal = np.tile(signal, nbre_tirs)

    return lfm_signal, temps

def radar_get_matched_filter(waveform, win_type=1):
    nsamp = len(waveform)
    x = np.trim_zeros(waveform).reshape(1, -1)

    n = x.shape[0]
    if n > 1:
        x = np.conj(x).T

    N = x.shape[1]
    if win_type == 1:
        win = windows.boxcar(N)
    elif win_type == 2:
        win = windows.hamming(N)
    elif win_type == 3:
        win = windows.chebwin(N, at=60)
    elif win_type == 4:
        win = windows.kaiser(N, beta=np.pi)
    elif win_type == 5:
        win = windows.blackman(N)

    win = win.reshape(1, -1)

    mfcoeff = np.conj(np.fliplr(x))
    if(MODE_CALCUL == "gpu"):
        mfcoeff = np.array(win)* mfcoeff
    else:
        mfcoeff = win * mfcoeff

    # Attention padding
    tmp = np.concatenate([np.zeros((1, nsamp - x.shape[1])), mfcoeff], axis=1)
    H = np.fft.fft(tmp, nsamp, axis=1)
    return mfcoeff, H, win


def generate_N_spaced_angles(min_angle=-60, max_angle=60, min_spacing=10, N=2):
    angles = []

    for _ in range(N):
        # Générer un angle aléatoire entre min_angle et max_angle
        angle = int(random.uniform(min_angle, max_angle))

        # S'assurer que l'angle généré est espacé d'au moins min_spacing des angles existants
        while any(abs(angle - a) < min_spacing for a in angles):
            angle = int(random.uniform(min_angle, max_angle))

        angles.append(angle)

    return np.array(angles)


def data_generation(doa, txSig_chan, N,d,SNR,H):
    theta = np.zeros(181)

    # Réception array
    sig_rx = receive_array(txSig_chan, N, d, doa)[0]
    sigNoise = awgn_noise(sig_rx, SNR)

    # Compression
    X = matched_filter(sigNoise, H)  # replace sig_rx by sigNoise
    # Matrice de corrélation
    Rxx = (1 / X.shape[1]) * np.dot(X, X.conj().T)
    data = np.concatenate([Rxx.real.flatten(), Rxx.imag.flatten()]).reshape(
        1, -1
    )
    theta[doa+90] = 1
    label = theta.reshape(1,-1)
    # Add SNR at last column
    label = np.hstack((label, [[SNR]]))

    return Rxx, data, label
def music_plot(Rxx, M):
    if MODE_CALCUL == "gpu":
        thetaM = nump.arange(-90, 91, 0.1)
    else:
        thetaM = np.arange(-90, 91, 0.1)
    Pmusic, EN = music_doa(Rxx, thetaM, M)
    if (MODE_CALCUL == "gpu"):
        Pmusic = Pmusic.get()
        Pmusiclog10 = nump.log10(Pmusic)
    else:
        Pmusiclog10 = np.log10(Pmusic)
    plt.figure()

    plt.plot(thetaM, 10 * Pmusiclog10)
    plt.grid(True)
    plt.title("DOAs cibles")
    plt.xlabel("DOA (degree)")
    plt.ylabel("|P(theta)|")
    plt.show()
if __name__ == "__main__":
    fc = 2.725e9 #fréquence centrale (Hz)
    BW = 10e6   #largeur de bande (Hz)
    Te = 1.4e-3 #Durée de modulation (s) (Pulse width)
    T = 4.1e-3 #Période de répétition des impulsions (s)
    fs = 4 * BW #% Fréquence d'échantillonnage (Hz)
    c = c
    lambda_ = c / fc
    N = 10
    d = 0.5
    Dataset_X = pd.DataFrame()
    Dataset_y = pd.DataFrame()


    # Génération de l'onde LFM
    LFMDir = "Up"
    LFM, temps = lfm_waveform(BW, Te, fs, LFMDir, T)
    txSig = LFM.reshape(1, LFM.shape[0])
    # Radar Matched Filter
    mf_coeff, H, winTemp = radar_get_matched_filter(LFM)
    start_time = time.time()
    rayons = [10, 17, 25]
    for SNR in range(-5, 30, 1):
        # Canal radar en espace libre
        theta = np.arange(-90, 91, 1)
        for i in range(2000):
            M = 2#random.randint(2,3)  # Nombre de cible
            R = np.array([[rayons[i]-random.randint(0,2)] for i in range(M)])
            txSig_chan = radar_channel(txSig, fc, fs, R, two_way=True)
            sigPower = np.sum(np.abs(txSig_chan) ** 2, axis=1) / txSig_chan.shape[1]
            sigPower = np.sqrt(sigPower[0] / sigPower[1:])[0]
            txSig_chan[1:, :] *= sigPower
            # SNR = random.randint(-5, 30)

            doa = generate_N_spaced_angles(min_angle=-60, max_angle=60, min_spacing=5, N=M)
            Rxx, data, label = data_generation(doa, txSig_chan, N, d, SNR, H)
            #music_plot(Rxx,M)
            if MODE_CALCUL == "gpu":
                Dataset_X = pd.concat(
                    [Dataset_X, pd.DataFrame(data.get())],
                    ignore_index=True,
                )
                Dataset_y = pd.concat(
                    [
                        Dataset_y,
                        pd.DataFrame(label.get()),
                    ],
                    ignore_index=True,
                )
            else:
                Dataset_X = pd.concat(
                    [Dataset_X, pd.DataFrame(data)],
                    ignore_index=True,
                )
                Dataset_y = pd.concat(
                    [
                        Dataset_y,
                        pd.DataFrame(label),
                    ],
                    ignore_index=True,
                )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Le code a pris {execution_time} secondes pour s'exécuter.")

    # persistence
    randomfileid = random.randint(0, 10000)
    Dataset_X.to_csv(f'Dataset_X{randomfileid}_30-5_3S.csv', index=False)
    Dataset_y.to_csv(f'Dataset_y{randomfileid}_30-5_3S.csv', index=False)

    #MUSIC verification
    music_plot(Rxx,M)

