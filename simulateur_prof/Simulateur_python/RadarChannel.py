from simulateur_prof.Simulateur_python.sim_utils import get_mode_calcul

if get_mode_calcul() == "gpu":
    import cupy as np
else:
    import numpy as np
from scipy.constants import c


def radar_channel(x, fc, fs, R, two_way=True, v=0):
    """
    Simule le canal radar en espace libre.

    Parameters:
    - x: Signal d'entrée
    - fc: Fréquence centrale (Hz)
    - fs: Fréquence d'échantillonnage (Hz)
    - R: Distance entre le radar et la cible (peut être une matrice [x, y])
    - two_way: Indique s'il s'agit d'un aller-retour (True) ou seulement un aller simple (False)
    - v: Vitesse de la cible (m/s)

    Returns:
    - y: Signal de sortie après le canal radar
    """

    Ts = 1 / fs
    wavelength = c / fc

    if R.shape[1] == 2:
        xd, yd = R[:, 0], R[:, 1]
        R = np.sqrt(xd**2 + yd**2)
        an = np.cos(np.arctan2(yd, xd))
    else:
        R = R[:, 0]
        an = 1

    if two_way:
        t = 2 * R / c
        fs_loss = wavelength**2 / (4 * np.pi * R) ** 2
        fs_loss = fs_loss**2
    else:
        t = R / c
        fs_loss = wavelength**2 / (4 * np.pi * R) ** 2

    Ndelay = t / Ts
    N = np.round(Ndelay)
    frac_delay = Ndelay - N

    w = np.sqrt(fs_loss) * np.exp(-1j * 2 * np.pi * 2 * R / wavelength)

    y = np.zeros((len(N), x.shape[1]), dtype=np.complex128)

    if np.all(N < x.shape[1]):
        for i in range(len(N)):
            if len(N) == x.shape[0]:
                y[i, :] = np.concatenate(
                    [np.zeros(int(N[i])), w[i] * x[i, : int(-N[i])]]
                )
            elif x.shape[0] == 1:
                y[i, :] = np.concatenate(
                    [np.zeros(int(N[i])), w[i] * x[0, : int(-N[i])]]
                )
            else:
                raise ValueError("Error!")

        if v != 0:
            v = v * an
            if two_way:
                fd = 2 * v / wavelength
            else:
                fd = v / wavelength
            mfd = fd * np.arange(len(y))
            Av = np.exp(-1j * 2 * np.pi * mfd * Ts)
            y = Av * y
    else:
        raise ValueError(f"Error: Rmax = {round(len(x) * Ts * c / 2 * 1e-3, 2)} Km")

    return y
