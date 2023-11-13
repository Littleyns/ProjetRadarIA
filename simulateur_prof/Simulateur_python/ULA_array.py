from utils import get_mode_calcul


if get_mode_calcul() == "gpu":
    import cupy as np
else:
    import numpy as np

from scipy.signal import hamming, kaiser, blackman


def ula_array(N, d, theta, win_type=1):
    """
    Uniform Linear Array

    Parameters:
    - N: Nombre d'éléments
    - d: Espacement des éléments en longueur d'onde
    - theta: Direction du faisceau en degré
    - win_type: Type de fenêtre (1: rectangulaire, 2: hamming, 3: hanning, 4: kaiser, 5: blackman)

    Returns:
    - A: Matrice du réseau d'antennes
    """

    n = np.arange(N) - (N - 1) / 2

    if win_type == 1:
        win = np.ones(N)
    elif win_type == 2:
        win = hamming(N)
    elif win_type == 4:
        win = kaiser(N, np.pi)
    elif win_type == 5:
        win = blackman(N)

    B = np.exp(
        1j * 2 * np.pi * n.reshape(1, 10).transpose() * d * np.sin(np.radians(theta))
    )
    A = win.reshape(1, 10).transpose() * B

    return A
