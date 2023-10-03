
import numpy as np
from sklearn.metrics import mean_squared_error

from Evaluation.Evaluateur import Evaluateur

class MSEEvaluateur(Evaluateur):
    def __init__(self):
        pass
    def evaluate(self , y_true, y_pred):
        # Calcul du MSE
        mse = mean_squared_error(y_true, y_pred)
        print("Mean square error: " + str(mse))

class RMSEEvaluateur(Evaluateur):
    def __init__(self):
        pass
    def evaluate(self, y_true, y_pred):
        # Calcul du MSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print("Root Mean square error: " + str(rmse))