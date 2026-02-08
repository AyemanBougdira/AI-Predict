import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
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


# -----------------------------
# Sidebar : paramètres + débits
# -----------------------------
with st.sidebar:
    st.header("Paramètres du four (communs)")
    T_initial = st.number_input("Température initiale (°C)", value=25.0)
    capacite_thermique = st.number_input("Capacité thermique (kJ/°C)", value=5000.0)
    temps_simulation = st.number_input("Durée simulation (h)", value=5.0)
    temps = np.linspace(0, temps_simulation, 150)

    st.header("Débits des 4 combustibles (communs)")
    n = 4
    noms = []
    masses = []
    is_alt_list = []

    for i in range(n):
        st.subheader(f"Combustible {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            nom = st.text_input(
                f"Nom combustible {i+1}", value=f"Combustible {i+1}", key=f"nom_{i}"
            )
        with col2:
            masse = st.number_input(
                f"Masse {i+1} (kg/h)", value=50.0, min_value=0.0, key=f"masse_{i}"
            )

        # Pour KPI TSR
        is_alt = st.checkbox("Combustible alternatif ?", value=True, key=f"is_alt_{i}")

        noms.append(nom)
        masses.append(masse)
        is_alt_list.append(is_alt)

    debit_total = float(np.sum(masses))


# -----------------------------
# Onglets
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1) Sans IA",
    "2) Avec IA (ML)",
    "3) Modifier le modele",
    "4) KPIs (Clinker)"
])

# ============================
# Variables partagées (pour Tab4)
# ============================
# Sans IA
energie_min = energie_max = None
T_min = T_max = None

# Avec IA
use_ml = True
energie_ml_min = energie_ml_max = None
T_ml_min = T_ml_max = None
pci_ml_min_eq = pci_ml_max_eq = None
largeur_ml = None

# Pour TSR (avec IA)
combustibles = []  # contiendra pci_min_pred/pci_max_pred, masse, is_alt

# =========================================================
# 1) SANS IA : Intervalle global [PCImin, PCImax]
# =========================================================
with tab1:
    st.subheader("Cas 1 — Sans IA (intervalle global)")
    st.write("Tu fixes un intervalle global de PCI (incertitude large).")

    pci_min_init = st.number_input("PCI min (kJ/kg)", value=15000.0, key="pci_min_init")
    pci_max_init = st.number_input("PCI max (kJ/kg)", value=30000.0, key="pci_max_init")

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
        ax.fill_between(temps, T_min, T_max, alpha=0.30, label="Zone probable (sans IA)")
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

    use_ml = st.checkbox("Activer l'IA", value=True, key="use_ml")

    if not use_ml:
        st.info("IA désactivée.")
    else:
        try:
            model_min, model_max = load_models()
        except Exception:
            st.error("Impossible de charger les modèles .joblib. Vérifie pci_min_model.joblib / pci_max_model.joblib.")
            st.stop()

        st.markdown("### Caractéristiques mesurables")
        X_list = []

        for i in range(n):
            st.markdown(f"**{noms[i]}**")
            cols = st.columns(4)
            hum = cols[0].number_input("Humidité (%)", 0.0, 100.0, 10.0, key=f"hum_{i}")
            c = cols[1].number_input("C (%)", 0.0, 100.0, 50.0, key=f"c_{i}")
            h = cols[2].number_input("H (%)", 0.0, 100.0, 6.0, key=f"h_{i}")
            ash = cols[3].number_input("Cendres (%)", 0.0, 100.0, 5.0, key=f"ash_{i}")

            X_list.append(np.array([[hum, c, h, ash]], dtype=float))

        if debit_total <= 0:
            st.warning("Débit total nul : augmente au moins une masse (kg/h).")
        else:
            energie_ml_min = 0.0
            energie_ml_max = 0.0
            combustibles = []

            st.markdown("### Intervalles PCI prédits")
            for i in range(n):
                pci_min_pred = float(model_min.predict(X_list[i])[0])
                pci_max_pred = float(model_max.predict(X_list[i])[0])

                if pci_min_pred > pci_max_pred:
                    pci_min_pred, pci_max_pred = pci_max_pred, pci_min_pred

                energie_ml_min += pci_min_pred * masses[i]
                energie_ml_max += pci_max_pred * masses[i]

                combustibles.append({
                    "nom": noms[i],
                    "masse": masses[i],
                    "is_alt": is_alt_list[i],
                    "pci_min_pred": pci_min_pred,
                    "pci_max_pred": pci_max_pred
                })

                st.write(
                    f"- {noms[i]} | masse={masses[i]:.2f} kg/h | "
                    f"PCI ∈ **[{pci_min_pred:.0f}, {pci_max_pred:.0f}] kJ/kg**"
                )

            T_ml_min = T_initial + (energie_ml_min / capacite_thermique) * temps
            T_ml_max = T_initial + (energie_ml_max / capacite_thermique) * temps

            pci_ml_min_eq = energie_ml_min / max(debit_total, 1e-9)
            pci_ml_max_eq = energie_ml_max / max(debit_total, 1e-9)
            largeur_ml = pci_ml_max_eq - pci_ml_min_eq

            st.markdown("### Résultats globaux (IA)")
            st.write(f"PCI équivalent global ∈ **[{pci_ml_min_eq:.0f}, {pci_ml_max_eq:.0f}] kJ/kg**")
            st.write(f"Largeur zone (IA) : **{largeur_ml:.0f} kJ/kg**")
            st.write(f"Température finale min : **{T_ml_min[-1]:.2f} °C**")
            st.write(f"Température finale max : **{T_ml_max[-1]:.2f} °C**")

            fig2, ax2 = plt.subplots()
            ax2.plot(temps, T_ml_min, label="PCI min (ML)")
            ax2.plot(temps, T_ml_max, label="PCI max (ML)")
            ax2.fill_between(temps, T_ml_min, T_ml_max, alpha=0.35, label="Zone probable (avec IA)")
            ax2.set_title("Avec IA — Zone probable réduite")
            ax2.set_xlabel("Temps (h)")
            ax2.set_ylabel("Température (°C)")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

# =========================================================
# 3) Modifier le modèle : base + ajout dataset + réentrainement
# =========================================================
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
        needed = ["humidite","carbone","hydrogene","cendres","pci_min","pci_max"]
        df = df.copy()
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")

        for c in needed:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=needed)

        swap = df["pci_min"] > df["pci_max"]
        if swap.any():
            df.loc[swap, ["pci_min","pci_max"]] = df.loc[swap, ["pci_max","pci_min"]].values

        df = df[
            df["humidite"].between(0,100) &
            df["carbone"].between(0,100) &
            df["hydrogene"].between(0,100) &
            df["cendres"].between(0,100)
        ].reset_index(drop=True)

        return df

    df_train = load_training_data(DATA_PATH)
    try:
        df_train = clean_df(df_train) if len(df_train) else df_train
    except Exception as e:
        st.error(f"Dataset existant invalide: {e}")
        st.stop()

    tabX, tabY = st.tabs(["Base actuelle", "Ajouter un nouveau dataset"])

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
            try:
                new_df = pd.read_csv(uploaded)
                new_df = clean_df(new_df)
            except Exception as e:
                st.error(f"Dataset importé invalide: {e}")
                st.stop()

            st.write("Aperçu dataset importé :")
            st.dataframe(new_df.head(20), use_container_width=True)

            if st.button("Valider l'ajout du dataset"):
                if merge_mode == "Remplacer la base":
                    df_train = new_df.copy()
                else:
                    df_train = pd.concat([df_train, new_df], ignore_index=True)

                df_train = clean_df(df_train)
                df_train.to_csv(DATA_PATH, index=False)
                st.cache_data.clear()
                st.success(f"Base mise à jour. Total lignes : {len(df_train)}")

    st.markdown("---")
    st.markdown("## Ré-entraîner les modèles (pci_min / pci_max)")

    def make_model():
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                # pour limiter taille (optionnel)
                max_depth=14,
                min_samples_leaf=2
            ))
        ])

    if st.button("Ré-entraîner et sauvegarder les modèles"):
        if len(df_train) < 50:
            st.error("Dataset trop petit pour entraîner un modèle (ajoute plus de lignes).")
        else:
            X = df_train[["humidite","carbone","hydrogene","cendres"]]
            y_min = df_train["pci_min"]
            y_max = df_train["pci_max"]

            model_min_new = make_model()
            model_max_new = make_model()

            model_min_new.fit(X, y_min)
            model_max_new.fit(X, y_max)

            joblib.dump(model_min_new, MODEL_MIN_PATH, compress=3)
            joblib.dump(model_max_new, MODEL_MAX_PATH, compress=3)

            st.cache_resource.clear()
            st.success("Modèles ré-entraînés et sauvegardés ✅")

# =========================================================
# 4) KPIs — Clinker
# =========================================================
with tab4:
    st.header("KPIs — Cimenterie / Clinker")

    if debit_total <= 0:
        st.warning("Débit total nul : configure les combustibles dans la sidebar.")
        st.stop()

    st.subheader("Entrées procédé")
    prod_clinker_tph = st.number_input("Production clinker (t/h)", value=150.0, min_value=0.0, key="prod_clinker_tph")
    eta = st.slider("Rendement global η (combustion + transfert)", 0.1, 1.0, 0.75, 0.01, key="eta")
    Tmin_cible = st.number_input("T min cible (°C)", value=1300.0, key="Tmin_cible")
    Tmax_cible = st.number_input("T max cible (°C)", value=1500.0, key="Tmax_cible")

    def risk_label(Tmin_curve, Tmax_curve):
        if Tmin_curve is None or Tmax_curve is None:
            return "Non calculé (exécute d'abord Tab1/Tab2)"
        if float(np.max(Tmax_curve)) < Tmin_cible:
            return "Risque élevé de sous-cuisson (zone trop basse)"
        if float(np.min(Tmin_curve)) > Tmax_cible:
            return "Risque élevé de surchauffe (zone trop haute)"
        if float(np.min(Tmin_curve)) < Tmin_cible and float(np.max(Tmax_curve)) > Tmax_cible:
            return "Risque mixte (incertitude traverse la plage)"
        if float(np.max(Tmax_curve)) > Tmax_cible:
            return "Risque de surchauffe possible"
        if float(np.min(Tmin_curve)) < Tmin_cible:
            return "Risque de sous-cuisson possible"
        return "OK (zone compatible avec la plage)"

    # --------- KPIs Sans IA ----------
    st.markdown("### KPIs — Sans IA")
    if energie_min is None or energie_max is None or T_min is None or T_max is None:
        st.info("Va dans l’onglet **1) Sans IA** pour calculer la zone, puis reviens ici.")
    else:
        pci_eq_init_min = energie_min / max(debit_total, 1e-9)
        pci_eq_init_max = energie_max / max(debit_total, 1e-9)

        st.write(f"Énergie totale ∈ **[{energie_min:.0f}, {energie_max:.0f}] kJ/h**")
        st.write(f"PCI équivalent global ∈ **[{pci_eq_init_min:.0f}, {pci_eq_init_max:.0f}] kJ/kg**")
        st.write(f"Énergie utile (η) ∈ **[{eta*energie_min:.0f}, {eta*energie_max:.0f}] kJ/h**")
        st.write(f"Diagnostic plage T : **{risk_label(T_min, T_max)}**")

        if prod_clinker_tph > 0:
            # kJ/kg clinker = MJ/t clinker (même valeur numérique)
            spec_min = energie_min / (prod_clinker_tph * 1000.0)
            spec_max = energie_max / (prod_clinker_tph * 1000.0)
            st.write(f"Consommation thermique spécifique ∈ **[{spec_min:.0f}, {spec_max:.0f}] kJ/kg clinker**")
            st.write(f"(équivalent) ∈ **[{spec_min:.1f}, {spec_max:.1f}] MJ/t clinker**")
        else:
            st.info("Production clinker = 0 → KPI consommation spécifique non calculé.")

    st.markdown("---")

    # --------- KPIs Avec IA ----------
    st.markdown("### KPIs — Avec IA (ML)")
    if (not use_ml) or (energie_ml_min is None) or (energie_ml_max is None) or (T_ml_min is None) or (T_ml_max is None):
        st.info("Active l'IA et calcule dans l’onglet **2) Avec IA (ML)**, puis reviens ici.")
    else:
        pci_ml_min_eq = energie_ml_min / max(debit_total, 1e-9)
        pci_ml_max_eq = energie_ml_max / max(debit_total, 1e-9)

        st.write(f"Énergie totale ∈ **[{energie_ml_min:.0f}, {energie_ml_max:.0f}] kJ/h**")
        st.write(f"PCI équivalent global ∈ **[{pci_ml_min_eq:.0f}, {pci_ml_max_eq:.0f}] kJ/kg**")
        st.write(f"Énergie utile (η) ∈ **[{eta*energie_ml_min:.0f}, {eta*energie_ml_max:.0f}] kJ/h**")
        st.write(f"Diagnostic plage T : **{risk_label(T_ml_min, T_ml_max)}**")

        if prod_clinker_tph > 0:
            spec_min = energie_ml_min / (prod_clinker_tph * 1000.0)
            spec_max = energie_ml_max / (prod_clinker_tph * 1000.0)
            st.write(f"Consommation thermique spécifique ∈ **[{spec_min:.0f}, {spec_max:.0f}] kJ/kg clinker**")
            st.write(f"(équivalent) ∈ **[{spec_min:.1f}, {spec_max:.1f}] MJ/t clinker**")

        # TSR (Thermal Substitution Rate)
        Q_alt_min = 0.0
        Q_alt_max = 0.0
        for cb in combustibles:
            if cb.get("is_alt", False):
                Q_alt_min += cb["pci_min_pred"] * cb["masse"]
                Q_alt_max += cb["pci_max_pred"] * cb["masse"]

        tsr_min = Q_alt_min / max(energie_ml_min, 1e-9)
        tsr_max = Q_alt_max / max(energie_ml_max, 1e-9)
        st.write(f"TSR (substitution thermique) ∈ **[{100*tsr_min:.1f}%, {100*tsr_max:.1f}%]**")
