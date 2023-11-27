import matplotlib.pyplot as plt
import numpy as np
from Evaluation.Evaluateur import Evaluateur


class PredictedStepPlot(Evaluateur):
    def evaluate(self, y_true, y_pred, threshold) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Resultat attendu vs resultat prédit')
        ax1.set_title("attendu")
        ax1.step(np.arange(0, y_pred.shape[0], 1), y_true)
        ax2.set_title("Predit")
        ax2.step(np.arange(0, y_pred.shape[0], 1), y_pred, color="darkorange")
        ax2.axhline(y=threshold, color='red', linestyle='--', label='seuil binaire')
        plt.figure()
        plt.title("Cote à cote")
        plt.step(np.arange(0, y_pred.shape[0], 1),y_true)
        plt.step(np.arange(0, y_pred.shape[0], 1), y_pred)

class LearningCurvesPlot(Evaluateur):
    def evaluate(self, history):
        plt.figure(figsize=(12, 4))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.show()