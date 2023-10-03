import pandas as pd
import numpy as np
from Data.RadarDataSet import RadarDataSet


class DataLoader:
    def __init__(self):
        self.dataPath = "./Data/Data_Doa.csv"
        self.dataLabelsPath = "./Data/Label_Doa.csv"

    def load_data(self):
        data = pd.read_csv(self.dataPath)
        labels = pd.read_csv(self.dataLabelsPath)

        data = np.array(data).transpose()
        labels = np.array(labels).transpose()
        #loaded_data = []
        #for i in range(data.shape[1]):
        #    loaded_data += RadarData.RadarData(data.iloc[:, [i]], labels.iloc[:, [i]]),
        #adar_dataset = RadarDataSet(data, labels, loaded_data)

        return data, labels


    # Ajoutez d'autres méthodes de gestion des données
