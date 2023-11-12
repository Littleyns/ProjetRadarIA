import numpy as np


def music_doa(Rxx, theta, M):
    """
    Calcule les angles d'arrivées des signaux RF non-corrélées en présence de bruit additif gaussien.

    Parameters:
    - Rxx: Matrice d'auto-corrélation du réseau entier
    - theta: Liste d'angles d'arrivées des signaux
    - M: Nombre de signaux

    Returns:
    - Pmusic: Pseudo-spectre MUSIC
    - EN: Sous-espace bruit
    """

    # Décomposition en vecteurs singuliers
    Dia, V = np.linalg.eig(Rxx)

    # Trie des valeurs propres par ordre croissant
    Index = np.argsort(Dia)

    # Calcul de la matrice des (N-M) vecteurs propres associés au sous-espace bruit
    EN = V[:, Index[: len(Rxx) - M]]

    # Calcul du pseudo-spectre MUSIC
    Pmusic = np.zeros(len(theta))
    n = np.arange(1, len(Rxx) + 1).reshape(10, 1)

    for k in range(len(theta)):
        a = np.exp(1j * (n - 1) * np.pi * np.sin(np.radians(theta[k])))
        Pmusic[k] = 1 / np.abs(
            np.dot(a.conj().transpose(), EN) @ np.dot(EN.conj().transpose(), a)
        )  # Pseudo-spectre MUSIC

    return Pmusic, EN
