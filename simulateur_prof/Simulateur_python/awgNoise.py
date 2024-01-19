from simulateur_prof.Simulateur_python.sim_utils import get_mode_calcul
if get_mode_calcul() == "gpu":
    import cupy as np
else:
    import numpy as np
def awgn_noise(sig, req_snr):
    """
    Ajoute du bruit gaussien blanc à un signal.

    Parameters:
    - sig: Signal d'entrée
    - req_snr: Rapport signal-sur-bruit (SNR) requis en décibels

    Returns:
    - y: Signal avec bruit gaussien blanc ajouté
    """

    sig_power = np.sum(np.abs(sig.flatten()) ** 2) / np.size(sig)

    req_snr = 10 ** (req_snr / 10)
    noise_power = sig_power / req_snr

    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)
    )
    y = sig + noise

    return y
