from typing import List

import numpy as np
from Data import RadarData

def ajout_signal_frequentiel(radar_data: List[RadarData.RadarData]):
    for d in radar_data:
        signal_frequentiel = np.fft.fft(np.array(d.signal_domaine_temporel)[:, 0])
        d.setSignalDomaineFrequentiel(signal_frequentiel)
