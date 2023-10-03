from typing import List

import numpy as np

def get_signal_frequentiel(signal):
    signal_frequentiel = np.fft.fft(signal)
    return signal_frequentiel
