import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

st.title("Stabilité thermique du four — LafargeHolcim - ECC")

# -----------------------------
# Chargement des modèles (IA)
# -----------------------------


@st.cache_resource
def load_models():
    mmin = joblib.load("pci_min_model.joblib")
    mmax = joblib.load("pci_max_model.joblib")
    return mmin, mmax


with st.sidebar:
# -----------------------------
# Paramètres communs
# -----------------------------
    st.header("Paramètres du four (communs)")
    T_initial = st.number_input("Température initiale (°C)", value=25.0)
    capacite_thermique = st.number_input(
        "Capacité thermique (kJ/°C)", value=5000.0)
    temps_simulation = st.number_input("Durée simulation (h)", value=5.0)

    temps = np.linspace(0, temps_simulation, 150)

# -----------------------------
# Saisie des combustibles (commune)
# -----------------------------
    st.header("Débits des 4 combustibles (communs)")
    n = 4
    noms = []
    masses = []

    for i in range(n):
        col1, col2 = st.columns(2)
        with col1:
            nom = st.text_input(
                f"Nom combustible {i+1}", value=f"Combustible {i+1}", key=f"nom_{i}")
        with col2:
            masse = st.number_input(
                f"Masse {i+1} (kg/h)", value=50.0, min_value=0.0, key=f"masse_{i}")
        noms.append(nom)
        masses.append(masse)

    debit_total = float(np.sum(masses))

# -----------------------------
# Onglets : Sans IA / Avec IA
# -----------------------------
tab1, tab2, tab3 = st.tabs(["1) Sans IA", "2) Avec IA (ML)", "3) Modifier le modele "])

# =========================================================
# 1) SANS IA : Intervalle global [PCImin, PCImax]
# =========================================================
with tab1:
    st.subheader("Cas 1 — Sans IA (intervalle global)")
    st.write("Tu fixes un intervalle global de PCI (incertitude large).")

    pci_min_init = st.number_input(
        "PCI min (kJ/kg)", value=15000.0, key="pci_min_init")
    pci_max_init = st.number_input(
        "PCI max (kJ/kg)", value=30000.0, key="pci_max_init")

    if pci_min_init > pci_max_init:
        st.error("PCI min doit être ≤ PCI max.")
    elif debit_total <= 0:
        st.warning("Débit total nul : augmente au moins une masse (kg/h).")
    else:
        energie_min = pci_min_init * debit_total
        energie_max = pci_max_init * debit_total

        T_min = T_initial + (energie_min / capacite_thermique) * temps
        T_max = T_initial + (energie_max / capacite_thermique) * temps

        st.write(f"Débit total : **{debit_total:.2f} kg/h**")
        st.write(f"Température finale min : **{T_min[-1]:.2f} °C**")
        st.write(f"Température finale max : **{T_max[-1]:.2f} °C**")

        fig, ax = plt.subplots()
        ax.plot(temps, T_min, label="PCI min")
        ax.plot(temps, T_max, label="PCI max")
        ax.fill_between(temps, T_min, T_max, alpha=0.30,
                        label="Zone probable (sans IA)")
        ax.set_title("Sans IA — Zone probable large")
        ax.set_xlabel("Temps (h)")
        ax.set_ylabel("Température (°C)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

# =========================================================
# 2) AVEC IA : Intervalle par combustible [PCImin_i, PCImax_i]
# =========================================================
with tab2:
    st.subheader("Cas 2 — Avec IA (ML) : intervalle réduit")
    st.write("Le modèle ML prédit un intervalle PCI pour chaque combustible à partir des caractéristiques mesurables.")

    # Charger modèles ici (uniquement dans l'onglet IA)
    try:
        model_min, model_max = load_models()
    except Exception as e:
        st.error("Impossible de charger les modèles .joblib. Vérifie les fichiers pci_min_model.joblib / pci_max_model.joblib.")
        st.stop()

    # Caractéristiques par combustible
    st.markdown("### Caractéristiques mesurables")
    X_list = []

    for i in range(n):
        st.markdown(f"**{noms[i]}**")
        cols = st.columns(4)
        hum = cols[0].number_input(
            "Humidité (%)", 0.0, 100.0, 10.0, key=f"hum_{i}")
        c = cols[1].number_input("C (%)", 0.0, 100.0, 50.0, key=f"c_{i}")
        h = cols[2].number_input("H (%)", 0.0, 100.0, 6.0, key=f"h_{i}")
        ash = cols[3].number_input(
            "Cendres (%)", 0.0, 100.0, 5.0, key=f"ash_{i}")

        X_list.append(np.array([[hum, c, h, ash]], dtype=float))

    if debit_total <= 0:
        st.warning("Débit total nul : augmente au moins une masse (kg/h).")
    else:
        energie_ml_min = 0.0
        energie_ml_max = 0.0

        st.markdown("### Intervalles PCI prédits")
        for i in range(n):
            pci_min_pred = float(model_min.predict(X_list[i])[0])
            pci_max_pred = float(model_max.predict(X_list[i])[0])

            # sécurité : garantir min <= max
            if pci_min_pred > pci_max_pred:
                pci_min_pred, pci_max_pred = pci_max_pred, pci_min_pred

            energie_ml_min += pci_min_pred * masses[i]
            energie_ml_max += pci_max_pred * masses[i]

            st.write(
                f"- {noms[i]} | masse={masses[i]:.2f} kg/h | "
                f"PCI ∈ **[{pci_min_pred:.0f}, {pci_max_pred:.0f}] kJ/kg**"
            )

        T_ml_min = T_initial + (energie_ml_min / capacite_thermique) * temps
        T_ml_max = T_initial + (energie_ml_max / capacite_thermique) * temps

        # intervalle PCI "équivalent" global (pour mesurer la réduction)
        pci_ml_min_eq = energie_ml_min / debit_total
        pci_ml_max_eq = energie_ml_max / debit_total
        largeur_ml = pci_ml_max_eq - pci_ml_min_eq

        st.markdown("### Résultats globaux (IA)")
        st.write(
            f"PCI équivalent global ∈ **[{pci_ml_min_eq:.0f}, {pci_ml_max_eq:.0f}] kJ/kg**")
        st.write(f"Largeur zone (IA) : **{largeur_ml:.0f} kJ/kg**")
        st.write(f"Température finale min : **{T_ml_min[-1]:.2f} °C**")
        st.write(f"Température finale max : **{T_ml_max[-1]:.2f} °C**")

        fig2, ax2 = plt.subplots()
        ax2.plot(temps, T_ml_min, label="PCI min (ML)")
        ax2.plot(temps, T_ml_max, label="PCI max (ML)")
        ax2.fill_between(temps, T_ml_min, T_ml_max, alpha=0.35,
                         label="Zone probable (avec IA)")
        ax2.set_title("Avec IA — Zone probable réduite")
        ax2.set_xlabel("Temps (h)")
        ax2.set_ylabel("Température (°C)")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        # -----------------------------
# IA — Gestion dataset & modèle
# -----------------------------
with tab3:
    st.markdown("## Gestion de la base de données (IA)")

    DATA_PATH = "data_pci_interval.csv"
    MODEL_MIN_PATH = "pci_min_model.joblib"
    MODEL_MAX_PATH = "pci_max_model.joblib"

    @st.cache_data
    def load_training_data(path: str) -> pd.DataFrame:
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame(columns=["humidite","carbone","hydrogene","cendres","pci_min","pci_max"])

    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        # garder seulement les colonnes nécessaires
        needed = ["humidite","carbone","hydrogene","cendres","pci_min","pci_max"]
        df = df.copy()
        df = df[[c for c in needed if c in df.columns]]

        # forcer numérique
        for c in needed:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # drop lignes invalides
        df = df.dropna(subset=["humidite","carbone","hydrogene","cendres","pci_min","pci_max"])

        # corriger min/max
        swap = df["pci_min"] > df["pci_max"]
        if swap.any():
            df.loc[swap, ["pci_min","pci_max"]] = df.loc[swap, ["pci_max","pci_min"]].values

        # bornes simples (optionnel)
        df = df[(df["humidite"].between(0,100)) &
                (df["carbone"].between(0,100)) &
                (df["hydrogene"].between(0,100)) &
                (df["cendres"].between(0,100))]

        return df.reset_index(drop=True)

    df_train = load_training_data(DATA_PATH)
    df_train = clean_df(df_train)

    tabX, tabY = st.tabs(["Base atuelle", "Ajouter un nouveau dataset"])
    with tabX:
        st.write(f"Lignes : **{len(df_train)}**")
        st.dataframe(df_train, use_container_width=True)

        csv_bytes = df_train.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger la base actuelle (CSV)",
            data=csv_bytes,
            file_name="data_pci_interval.csv",
            mime="text/csv",
        )

    with tabY:
        uploaded = st.file_uploader("Importer un CSV (mêmes colonnes)", type=["csv"])

        merge_mode = st.radio("Mode d'ajout", ["Ajouter (concaténer)", "Remplacer la base"], index=0)

        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            new_df = clean_df(new_df)

            st.write("Aperçu dataset importé :")
            st.dataframe(new_df.head(20), use_container_width=True)

            if st.button("Valider l'ajout du dataset"):
                if merge_mode == "Remplacer la base":
                    df_train = new_df.copy()
                else:
                    df_train = pd.concat([df_train, new_df], ignore_index=True)

                df_train = clean_df(df_train)
                df_train.to_csv(DATA_PATH, index=False)

                # IMPORTANT: vider cache pour voir la MAJ
                st.cache_data.clear()
                st.success(f"Base mise à jour. Total lignes : {len(df_train)}")

    st.markdown("---")
    st.markdown("## Ré-entraîner les modèles (pci_min / pci_max)")

    def make_model():
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            ))
        ])

    if st.button("Ré-entraîner et sauvegarder les modèles"):
        if len(df_train) < 50:
            st.error("Dataset trop petit pour entraîner un modèle (ajoute plus de lignes).")
        else:
            X = df_train[["humidite","carbone","hydrogene","cendres"]]
            y_min = df_train["pci_min"]
            y_max = df_train["pci_max"]

            model_min = make_model()
            model_max = make_model()

            model_min.fit(X, y_min)
            model_max.fit(X, y_max)

            joblib.dump(model_min, MODEL_MIN_PATH)
            joblib.dump(model_max, MODEL_MAX_PATH)

            # vider cache resource si tu caches load_models()
            st.cache_resource.clear()
            st.success("Modèles ré-entraînés et sauvegardés ✅")
