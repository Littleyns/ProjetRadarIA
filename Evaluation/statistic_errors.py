
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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

class R2Score(Evaluateur):
    def __init__(self):
        pass
    def evaluate(self, y_true, y_pred):
        # Calcul du MSE
        r2 = np.sqrt(r2_score(y_true, y_pred))
        print("R2 Score : " + str(r2))