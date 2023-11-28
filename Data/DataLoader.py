import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, datapath ,labelpath):
        self.dataPath = datapath
        self.dataLabelsPath = labelpath

    def load_data(self):
        data = pd.read_csv(self.dataPath, index_col=False)
        labels = pd.read_csv(self.dataLabelsPath, index_col=False)

        data = np.array(data)#.transpose()
        labels = np.array(labels)#.transpose()
        # loaded_data = []
        # for i in range(data.shape[1]):
        #    loaded_data += RadarData.RadarData(data.iloc[:, [i]], labels.iloc[:, [i]]),
        # adar_dataset = RadarDataSet(data, labels, loaded_data)

        return data, labels

    # Ajoutez d'autres méthodes de gestion des données

if __name__ == "__main__":
    data = pd.read_csv("Dataset_X6687.csv", index_col=False)
    labels = pd.read_csv("Dataset_y6687.csv", index_col=False)


#data1=reel data2=img
#alt
#result = np.vstack((row1, row2) for row1, row2 in zip(data1, data2))

#matrice2
#merged_data = np.array([np.concatenate((row1, row2)) for row1, row2 in zip(data1, data2)])
#result1 = merged_data.reshape(data1.shape[0], -1, data1.shape[1])