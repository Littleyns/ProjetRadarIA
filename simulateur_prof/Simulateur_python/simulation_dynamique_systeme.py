import numpy as np
import pandas as pd
import random
def generate_time_series_example(num_positions, num_time_steps, doa_indices, max_speed):
    # Générer une vitesse aléatoire pour chaque exemple
    initial_doa_indices = doa_indices
    speeds = []
    for i in range(len(doa_indices)):
        speed = (1,-1)[np.random.randint(0,2)==1]
        if speed == 1:
            speed*= random.randint(1, int((abs(180-doa_indices[i]))/num_time_steps))
        else:
            speed *= random.randint(1,int(doa_indices[i]/num_time_steps))
        speeds+=speed,
    # Normaliser la vitesse
    #normalized_speed = speed / np.mean(speed)

    # Initialiser une série temporelle de positions pour chaque exemple
    time_series_positions = []

    for _ in range(num_time_steps):
        # Générer un vecteur de positions initialisé à zéro
        positions = np.zeros(num_positions)

        # Placer des '1' aux indices spécifiés pour les DoA
        positions[list(doa_indices)] = 1

        # Ajouter la série temporelle de positions à la liste
        time_series_positions.append(positions)

        # Mettre à jour les indices des DoA en fonction de la vitesse
        doa_indices = np.array(doa_indices) + np.array(speeds)
    # Créer un dataframe pandas avec la série temporelle de positions et la vitesse associée

    time_series_positions= np.array(time_series_positions)
    # Créer la DataFrame
    df = pd.DataFrame({i: time_series_positions[i].reshape(1,181).tolist() for i in range(time_series_positions.shape[0])})

    speeds_res = np.zeros(num_positions)
    speeds_res[initial_doa_indices] = speeds
    df['Speed'] = speeds_res.reshape(1,181).tolist()
    return df

# Paramètres
num_positions = 181
num_time_steps = 5

max_speed = 10.0  # Vitesse maximale

all_examples = pd.DataFrame()

for example_idx in range(200):
    doa_indices = [random.randint(30,150), random.randint(30,150)]
    # Générer une série temporelle de 5 mesures associée à une vitesse
    time_series_example = generate_time_series_example(num_positions, num_time_steps, doa_indices, max_speed)



    # Ajouter la ligne à l'ensemble des exemples
    all_examples = pd.concat([all_examples, time_series_example], axis=0, ignore_index=True)

# Enregistrer toutes les séries temporelles dans un seul fichier CSV
all_examples.to_csv('../../Data/all_time_series_examples.csv', index=False)