import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

humidite = np.random.uniform(0, 50, N)
carbone = np.random.uniform(30, 85, N)
hydrogene = np.random.uniform(2, 8, N)
cendres = np.random.uniform(0, 40, N)

# PCI moyen (inconnu en pratique)
pci_central = (
    330 * carbone
    + 1400 * hydrogene
    - 120 * humidite
    - 100 * cendres
)

# Incertitude dépendante de la qualité du combustible
incertitude = (
    2500
    + 40 * humidite
    + 30 * cendres
)

pci_min = np.clip(pci_central - incertitude, 8000, None)
pci_max = np.clip(pci_central + incertitude, None, 35000)

df = pd.DataFrame({
    "humidite": humidite,
    "carbone": carbone,
    "hydrogene": hydrogene,
    "cendres": cendres,
    "pci_min": pci_min,
    "pci_max": pci_max
})

df.to_csv("data_pci_interval.csv", index=False)
