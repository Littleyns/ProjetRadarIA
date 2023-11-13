from utils import get_mode_calcul


if get_mode_calcul() == "gpu":
    import cupy as np
else:
    import numpy as np

def win_gen(window, N, *args):
    """
    Génération de la fenêtre.

    Parameters:
    - window: Type de fenêtre ('rect', 'blackman', 'hamming', 'hann', 'bartlett', 'riesz', 'blackman-harris',
                               'kaiser', 'chebwin')
    - N: Longueur de la fenêtre
    - *args: Paramètres additionnels selon le type de fenêtre

    Returns:
    - w: Fenêtre générée
    """

    n = np.arange(N)

    if window == "rect":
        w = np.ones(N)
    elif window == "blackman":
        w = (
            0.42
            - 0.5 * np.cos(2 * np.pi * n / (N - 1))
            + 0.08 * np.cos(4 * np.pi * n / (N - 1))
        )
    elif window == "hamming":
        alpha = 0.54
        beta = 1 - alpha
        w = alpha - beta * np.cos(2 * np.pi * n / (N - 1))
    elif window == "hann":
        w = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    elif window == "bartlett":
        w = 1 - 2 * np.abs(n - (N - 1) / 2) / (N - 1)
    elif window == "riesz":
        n = np.linspace(-N / 2, N / 2, N)
        w = 1 - np.abs(n / (N / 2)) ** 2
    elif window == "blackman-harris":
        a0 = 0.35875
        a1 = 0.48829
        a2 = 0.14128
        a3 = 0.01168
        w = (
            a0
            - a1 * np.cos(2 * np.pi * n / (N - 1))
            + a2 * np.cos(4 * np.pi * n / (N - 1))
            - a3 * np.cos(6 * np.pi * n / (N - 1))
        )
    elif window == "kaiser":
        beta = args[0]
        alpha = beta * np.sqrt(1 - ((n - (N - 1) / 2) / ((N - 1) / 2)) ** 2)
        w = approx_besseli(0, alpha) / approx_besseli(0, beta)
    elif window == "chebwin":
        A = args[0]
        r = 10 ** (A / 20)
        x0 = np.cosh(1 / (N - 1) * np.arccosh(r))

        k = np.arange(1, (N - 1) // 2 + 1)
        x = x0 * np.cos(k * np.pi / N)
        Tn = cheby_poly(x, N - 1)
        s = Tn * np.cos(2 * k[:, np.newaxis] * np.pi * (n - (N - 1) / 2) / N)
        w = (1 / N) * (r + 2 * np.sum(s, axis=0))

        w = w / np.max(w)

    return w


def approx_besseli(v, x):
    y = np.zeros_like(x)
    for k in range(31):
        y += ((x / 2) ** k / np.math.factorial(k + v)) ** 2
    return y


def cheby_poly(x, n):
    return np.where(
        np.abs(x) <= 1, np.cos(n * np.arccos(x)), np.cosh(n * np.arccosh(x))
    )
