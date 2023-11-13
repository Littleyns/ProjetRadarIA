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
import chainer
import pandas as pd

from utils import get_mode_calcul

chainer.print_runtime_info()
# Enregistrez le temps de début
start_time = time.time()
executor = ThreadPoolExecutor()
# report the number of worker threads chosen by default
print(executor._max_workers)
environ["OMP_NUM_THREADS"] = str(executor._max_workers)
MODE_CALCUL = get_mode_calcul()
print("MODE_CALCUL: " + MODE_CALCUL)
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
    if MODE_CALCUL == "gpu":
        mfcoeff = np.array(win) * mfcoeff
    else:
        mfcoeff = win * mfcoeff

    # Attention padding
    tmp = np.concatenate([np.zeros((1, nsamp - x.shape[1])), mfcoeff], axis=1)
    H = np.fft.fft(tmp, nsamp, axis=1)
    return mfcoeff, H, win


# def ula_array(N, d, theta):
#    return np.exp(1j * 2 * np.pi * d * np.arange(N) * np.sin(np.radians(theta)))


# Paramètres du système
fc = 2.725e9
BW = 10e6
Te = 1.4e-3
T = 4.1e-3
fs = 4 * BW
c = c
lambda_ = c / fc
N = 10
d = 0.5
SNR = 20
M = 2
R = np.array([[10], [20]])


# Génération de l'onde LFM
LFMDir = "Up"
LFM, temps = lfm_waveform(BW, Te, fs, LFMDir, T)
txSig = LFM.reshape(1, LFM.shape[0])

# Radar Matched Filter
mf_coeff, H, winTemp = radar_get_matched_filter(LFM)


# Canal radar en espace libre
txSig_chan = radar_channel(txSig, fc, fs, R, True)
sigPower = np.sum(np.abs(txSig_chan) ** 2, axis=1) / txSig_chan.shape[1]
sigPower = np.sqrt(sigPower[0] / sigPower[1:])[0]

temporaryVar1 = txSig_chan[1:, :]
txSig_chan[1:, :] *= sigPower

Dtheta = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
theta = np.arange(-90, 91, 1)
Dataset_X = pd.DataFrame()
Dataset_y = pd.DataFrame()

for j, dtheta in enumerate(Dtheta):
    theta1 = np.arange(
        -60, 61 - dtheta, 1
    )  # Redemander au prof l'explication de cette partie Dtheta
    theta2 = theta1 + dtheta

    data = np.zeros((200, len(theta1)))
    label = np.zeros((181, len(theta1)))
    for i, (doa1, doa2) in enumerate(zip(theta1, theta2)):
        doa = np.array([doa1, doa2])

        # Réception array
        sig_rx = receive_array(txSig_chan, N, d, doa)[0]
        sigNoise = awgn_noise(sig_rx, SNR)

        # Compression
        X = matched_filter(sigNoise, H)  # replace sig_rx by sigNoise
        # Matrice de corrélation
        Rxx = (1 / X.shape[1]) * np.dot(X, X.conj().T)
        data = np.concatenate([Rxx.real.flatten(), Rxx.imag.flatten()]).reshape(1, -1)
        label = ((theta == doa1) + (theta == doa2)).astype(int).reshape(1, -1)

        Dataset_X = pd.concat(
            [Dataset_X, (data, pd.DataFrame(data.get()))[MODE_CALCUL == "gpu"]],
            ignore_index=True,
        )
        Dataset_y = pd.concat(
            [Dataset_y, (label, pd.DataFrame(label.get()))[MODE_CALCUL == "gpu"]],
            ignore_index=True,
        )
# Affichage des résultats

thetaM = nump.arange(-90, 91, 0.1)
Pmusic, EN = music_doa(Rxx, thetaM, M)
if MODE_CALCUL == "gpu":
    Pmusic = Pmusic.get()
    Pmusiclog10 = nump.log10(Pmusic)
else:
    Pmusiclog10 = np.log10(Pmusic)
end_time = time.time()

# Calculez la durée totale
execution_time = end_time - start_time

print(f"Le code a pris {execution_time} secondes pour s'exécuter.")

plt.figure()

plt.plot(thetaM, 10 * Pmusiclog10)
plt.grid(True)
plt.title("DOAs cibles")
plt.xlabel("DOA (degree)")
plt.ylabel("|P(theta)|")
plt.show()
