import pandas as pd
import os
#Division des données par SNR
fileIdentifier="9193_3-2S_30-30"
X = pd.read_csv(f"./Dataset_X{fileIdentifier}.csv",index_col=False)
y= pd.read_csv(f"./Dataset_y{fileIdentifier}.csv",index_col=False)
y_snr = y.iloc[:, -1]
y = y.iloc[:,:-1]
df = pd.DataFrame({'X':  X.values.tolist(), 'y': y.values.tolist(), 'snr':y_snr.values.tolist()})

#conditions = lambda row: (np.sum(row['y']) == 2)  # Condition sur le nombre de sources
conditions = lambda row: (row['snr'] >=-5)  # Condition sur le SNR
# Filtrer le DataFrame en fonction des conditions
df_filtered = df[df.apply(conditions, axis=1)]

df_X = pd.DataFrame(df_filtered['X'].tolist(), index=df_filtered.index, columns=[i for i in range(len(df_filtered['X'].iloc[0]))])
df_y = pd.DataFrame(df_filtered['y'].tolist(), index=df_filtered.index, columns=[i for i in range(len(df_filtered['y'].iloc[0]))])
df_y[181]= df_filtered['snr']

if not os.path.exists(f'./dataset-{fileIdentifier}'):
    os.mkdir(f"./dataset-{fileIdentifier}")
# Enregistrer le DataFrame dans un fichier CSV
df_X.to_csv(f'./dataset-{fileIdentifier}/Dataset_X_30-5_3-2S.csv',header=True, index=False)
df_y.to_csv(f'./dataset-{fileIdentifier}/Dataset_y_30-5_3-2S.csv',header=True, index=False)