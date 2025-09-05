import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

st.set_page_config(page_title="Fibre Optique - Prédiction Retard", layout="centered",page_icon="⏳")

st.title("📡 Prédiction des Retards d’Installation Fibre Optique")
st.markdown("Remplissez le formulaire ci-dessous pour obtenir une prédiction.")

with st.form("prediction_form"):
    distance = st.number_input("Distance NRO (km)", min_value=0.0, max_value=100.0, value=5.0)
    nb_tech = st.number_input("Nombre de techniciens", min_value=1, max_value=20, value=3)
    longueur_fibre = st.number_input("Longueur fibre (m)", min_value=0, max_value=10000, value=500)
    duree_install = st.number_input("Durée installation (jours)", min_value=1, max_value=365, value=7)

    incident = st.selectbox("Incident signalé", ["Oui", "Non"])
    zone = st.selectbox("Zone", ["Casablanca", "Marrakech", "Rabat", "Tanger"])
    terrain = st.selectbox("Type de terrain", ["Rural", "Urbain"])
    meteo = st.selectbox("Météo", ["Pluie", "Soleil", "VentFort"])
    complexite = st.selectbox("Complexité chantier", ["Faible", "Moyenne", "Haute"])

    submitted = st.form_submit_button("🔮 Prédire")

if submitted:
    input_data = {col: 0 for col in X_columns}

    input_data["distance_NRO_km"] = distance
    input_data["nb_techniciens"] = nb_tech
    input_data["longueur_fibre_m"] = longueur_fibre
    input_data["duree_installation_jours"] = duree_install

    input_data["incident_signal"] = 1 if incident == "Oui" else 0

    if f"zone_{zone}" in input_data:
        input_data[f"zone_{zone}"] = 1

    if f"type_terrain_{terrain}" in input_data:
        input_data[f"type_terrain_{terrain}"] = 1

    if f"meteo_{meteo}" in input_data:
        input_data[f"meteo_{meteo}"] = 1

    if complexite != "Faible":  
        if f"complexite_chantier_{complexite}" in input_data:
            input_data[f"complexite_chantier_{complexite}"] = 1

    input_df = pd.DataFrame([input_data], columns=X_columns)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0].max()

    st.subheader("Résultat de la Prédiction")
    if prediction == 1:
        st.error(f"⚠️ Risque de **RETARD** avec une probabilité de {probability:.2f}")
    else:
        st.success(f"✅ **Pas de retard prévu** avec une probabilité de {probability:.2f}")
