
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score
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
    def evaluate(self, y_true, y_pred,Nsources=2, verbose=True):

        y_true_angles = np.sort(np.argsort(y_true, axis=1)[:, -Nsources:]-90,axis=1)
        y_pred_angles = np.sort(np.argsort(y_pred, axis=1)[:, -Nsources:]-90,axis=1)
        rmse = np.sqrt(mean_squared_error(y_true_angles, y_pred_angles))
        if verbose:
            print("Root Mean square error: " + str(rmse))
        return rmse

class R2Score(Evaluateur):
    def __init__(self):
        pass
    def evaluate(self, y_true, y_pred):
        # Calcul du MSE
        r2 = np.sqrt(r2_score(y_true, y_pred))
        print("R2 Score : " + str(r2))

class Accuracy(Evaluateur):
    def __init__(self):
        pass
    def evaluate(self, y_true, y_pred):
        # Calcul de l'accuracy
        acc = precision_score(y_true, y_pred, average='micro')
        print("Accuracy "+str(acc))
        return acc