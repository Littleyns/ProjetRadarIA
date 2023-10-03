import numpy as np
from sklearn.linear_model import LinearRegression



def linear_regression_doa(data, labels):
    # Entraînement d'un modèle de régression linéaire
    model = LinearRegression()

    model.fit(data, labels)

    return model