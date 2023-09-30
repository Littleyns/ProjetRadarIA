import pandas as pd
from Data import RadarData
class DataLoader:
    def __init__(self):
        self.dataPath = "./Data/Data_Doa.csv"
        self.dataLabelsPath = "./Data/Label_Doa.csv"

    def load_data(self):
        data = pd.read_csv(self.dataPath)
        labels = pd.read_csv(self.dataLabelsPath)
        loaded_data = []
        for i in range(data.shape[1]):
            loaded_data += RadarData.RadarData(data.iloc[:, [i]], labels.iloc[:, [i]]),
        return loaded_data


    # Ajoutez d'autres méthodes de gestion des données
