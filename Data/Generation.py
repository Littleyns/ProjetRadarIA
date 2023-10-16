import radarsimpy

# Création d'un transmetteur
transmitter = radarsimpy.Transmitter(
    f=10e9,  # Fréquence en Hz
    t=1e-6,  # Durée d'un pulse en secondes
    tx_power=30,  # Puissance de transmission en dBm
    pulses=100,  # Nombre de pulses
    prp=1e-3,  # Période de répétition des pulses en secondes
)

# Création d'un récepteur
receiver = radarsimpy.Receiver(
    fs=100e6,  # Taux d'échantillonnage en échantillons par seconde
    noise_figure=10,  # Facteur de bruit en dB
    rf_gain=20,  # Gain RF en dB
    load_resistor=50,  # Résistance de charge en Ohms
    baseband_gain=0,  # Gain en bande de base en dB
)

# Création d'un objet radar
radar = radarsimpy.Radar(transmitter, receiver)

# Simulation radar
radar.transmitter.si

# Récupération des données du radar
#data = radar.data
