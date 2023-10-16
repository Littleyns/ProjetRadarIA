import numpy as np
from scipy.interpolate import interp1d

def data_to_complex(data):
    data_im = data[:, 99:199]
    data_real = data[:, :100]
    return np.stack((data_im, data_real), axis=1)


def augmentDataInterp(data, new_dim):
    n_original = len(data[0])


    # Initialisez une matrice pour stocker les résultats interpolés
    matrice_interpolée = np.zeros((data.shape[0], new_dim))

    # Interpolation linéaire pour chaque ensemble de données
    for i in range(data.shape[0]):
        interpolator = interp1d(np.linspace(0, n_original - 1, n_original), data[i], kind='linear')
        indices_new = np.linspace(0, n_original - 1, new_dim)
        matrice_interpolée[i] = interpolator(indices_new)

    return matrice_interpolée