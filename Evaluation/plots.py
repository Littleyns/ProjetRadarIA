import matplotlib.pyplot as plt
import numpy as np
from Evaluation.Evaluateur import Evaluateur


class PredictedStepPlot(Evaluateur):
    def evaluate(self, y_true, y_pred) -> None:
        plt.step(np.arange(0, y_pred.shape[0], 1),y_true)
        plt.step(np.arange(0, y_pred.shape[0], 1), y_pred)