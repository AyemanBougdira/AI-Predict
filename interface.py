import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.title("Stabilité thermique du four — Réduction de l'incertitude PCI (ML)")


@st.cache_resource
def load_models():
    mmin = joblib.load("pci_min_model.joblib")
    mmax = joblib.load("pci_max_model.joblib")
    return mmin, mmax


model_min, model_max = load_models()

# --- Paramètres du four ---
st.header("Paramètres du four")
T_initial = st.number_input("Température initiale (°C)", value=25.0)
capacite_thermique = st.number_input(
    "Capacité thermique (kJ/°C)", value=5000.0)
temps_simulation = st.number_input("Durée simulation (h)", value=5.0)
temps = np.linspace(0, temps_simulation, 150)

# --- Intervalle initial (incertitude large) ---
st.header("Incertitude initiale (avant IA)")
pci_min_init = st.number_input("PCI min initial (kJ/kg)", value=15000.0)
pci_max_init = st.number_input("PCI max initial (kJ/kg)", value=30000.0)

# --- 4 combustibles ---
st.header("Combustibles (4) — saisie des caractéristiques")
n = 4
combustibles = []
for i in range(n):
    st.subheader(f"Combustible {i+1}")
    nom = st.text_input("Nom", value=f"Combustible {i+1}", key=f"nom_{i}")
    masse = st.number_input("Masse (kg/h)", value=50.0,
                            min_value=0.0, key=f"masse_{i}")

    hum = st.number_input("Humidité (%)", 0.0, 100.0, 10.0, key=f"hum_{i}")
    c = st.number_input("Carbone C (%)", 0.0, 100.0, 50.0, key=f"c_{i}")
    h = st.number_input("Hydrogène H (%)", 0.0, 100.0, 6.0, key=f"h_{i}")
    ash = st.number_input("Cendres (%)", 0.0, 100.0, 5.0, key=f"ash_{i}")

    X = np.array([[hum, c, h, ash]], dtype=float)

    combustibles.append({
        "nom": nom,
        "masse": masse,
        "X": X
    })

# --- Calcul zone initiale (large) ---
debit_total = sum(cb["masse"] for cb in combustibles)
energie_init_min = pci_min_init * debit_total
energie_init_max = pci_max_init * debit_total

T_init_min = T_initial + (energie_init_min / capacite_thermique) * temps
T_init_max = T_initial + (energie_init_max / capacite_thermique) * temps

# --- Calcul zone ML (réduite) ---
st.header("Réduction d'incertitude avec IA")
use_ml = st.checkbox("Activer l'IA pour réduire la zone probable", value=True)

if use_ml:
    energie_ml_min = 0.0
    energie_ml_max = 0.0

    st.subheader("PCI prédits (intervalle) par combustible")
    for cb in combustibles:
        pci_min_pred = float(model_min.predict(cb["X"])[0])
        pci_max_pred = float(model_max.predict(cb["X"])[0])

        # Garantir min <= max
        if pci_min_pred > pci_max_pred:
            pci_min_pred, pci_max_pred = pci_max_pred, pci_min_pred

        energie_ml_min += pci_min_pred * cb["masse"]
        energie_ml_max += pci_max_pred * cb["masse"]

        st.write(
            f"- **{cb['nom']}** | masse={cb['masse']:.2f} kg/h "
            f"| PCI ∈ [{pci_min_pred:.0f}, {pci_max_pred:.0f}] kJ/kg"
        )

    T_ml_min = T_initial + (energie_ml_min / capacite_thermique) * temps
    T_ml_max = T_initial + (energie_ml_max / capacite_thermique) * temps

    # Indicateur de réduction
    largeur_init = float(pci_max_init - pci_min_init)
    # Largeur “équivalente” globale ML en kJ/kg (ramenée au débit total)
    pci_ml_min_eq = energie_ml_min / max(debit_total, 1e-9)
    pci_ml_max_eq = energie_ml_max / max(debit_total, 1e-9)
    largeur_ml = float(pci_ml_max_eq - pci_ml_min_eq)

    st.subheader("Gain sur la zone probable")
    st.write(f"Intervalle initial: **{largeur_init:.0f} kJ/kg**")
    st.write(f"Intervalle ML (équivalent): **{largeur_ml:.0f} kJ/kg**")
    if largeur_init > 1e-9:
        st.write(f"Réduction: **{100*(1 - largeur_ml/largeur_init):.1f}%**")

    # --- Graphique ---
    st.subheader("Comparaison des zones")
    fig, ax = plt.subplots()

    ax.plot(temps, T_init_min, label="Zone initiale (PCI min)")
    ax.plot(temps, T_init_max, label="Zone initiale (PCI max)")
    ax.fill_between(temps, T_init_min, T_init_max,
                    alpha=0.20, label="Zone initiale (large)")

    ax.plot(temps, T_ml_min, label="Zone ML (PCI min)")
    ax.plot(temps, T_ml_max, label="Zone ML (PCI max)")
    ax.fill_between(temps, T_ml_min, T_ml_max,
                    alpha=0.35, label="Zone ML (réduite)")

    ax.set_xlabel("Temps (h)")
    ax.set_ylabel("Température (°C)")
    ax.set_title("Stabilité thermique — réduction de la zone probable par IA")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

else:
    st.info("IA désactivée — affichage uniquement de la zone initiale.")
    fig, ax = plt.subplots()
    ax.plot(temps, T_init_min, label="PCI min")
    ax.plot(temps, T_init_max, label="PCI max")
    ax.fill_between(temps, T_init_min, T_init_max,
                    alpha=0.3, label="Zone probable")
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel("Température (°C)")
    ax.set_title("Stabilité thermique avec PCI inconnu (zone initiale)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
