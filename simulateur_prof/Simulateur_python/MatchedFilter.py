import numpy as np
from scipy.fft import fft, ifft


def matched_filter(rxsig, h):
    """
    Applique un filtre adapté (Matched Filter) au signal reçu.

    Parameters:
    - rxsig: Signal reçu (peut être une matrice [lignes, échantillons])
    - h: Coefficients du filtre adapté (Matched Filter)

    Returns:
    - sig_pulse: Signal après le filtre adapté
    """

    l, _ = rxsig.shape
    h = h.reshape(1, -1)
    # Nombre d'échantillons dans le filtre
    nsamp = h.shape[1]

    if l > 1:
        rxsig = fft(rxsig, nsamp, axis=1)
        h = np.tile(h, (l, 1))
        sig_pulse = ifft(rxsig * h, axis=1)
    else:
        rxsig = fft(rxsig, nsamp)
        sig_pulse = ifft(rxsig * h)

    return sig_pulse
