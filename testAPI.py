import requests
import pandas as pd
import streamlit as st
from datetime import datetime

# Configuración de la API
API_FOOTBALL_KEY = "c0dabdb4909f07174443de8fa7e175fa"
API_FOOTBALL_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "X-RapidAPI-Key": API_FOOTBALL_KEY,
    "X-RapidAPI-Host": "v3.football.api-sports.io"
}

LEAGUE_ID = 140  # LaLiga
SEASON = 2024

# Función para obtener los equipos de la liga
@st.cache_data
def get_teams(league_id=LEAGUE_ID, season=SEASON):
    url = f"{API_FOOTBALL_URL}/teams?league={league_id}&season={season}"
    r = requests.get(url, headers=HEADERS).json()
    teams = {team['team']['name']: team['team']['id'] for team in r['response']}
    return teams

# Función para obtener los enfrentamientos históricos
def get_h2h(local_id, visitante_id):
    url = f"{API_FOOTBALL_URL}/fixtures?h2h={local_id}-{visitante_id}"
    r = requests.get(url, headers=HEADERS).json()
    matches = []
    for m in r['response']:
        match = m['fixture']
        home = m['teams']['home']['name']
        away = m['teams']['away']['name']
        date = datetime.fromisoformat(match['date'].replace('Z','')).strftime('%d-%m-%Y %H:%M')
        score = m['score']['fulltime']
        matches.append({
            "Fecha": date,
            "Local": home,
            "Visitante": away,
            "Goles Local": score['home'],
            "Goles Visitante": score['away'],
            "Estado": match['status']['short']
        })
    return pd.DataFrame(matches)

# Interfaz de usuario con Streamlit
st.title("📊 Historial de Enfrentamientos (H2H) - LaLiga 2024")

teams = get_teams()
team_names = list(teams.keys())

equipo_local = st.selectbox("Selecciona el equipo local", team_names)
equipo_visitante = st.selectbox("Selecciona el equipo visitante", team_names, index=1)

if st.button("Mostrar enfrentamientos"):
    if equipo_local == equipo_visitante:
        st.warning("Selecciona dos equipos diferentes.")
    else:
        st.info("Buscando datos…")
        local_id = teams[equipo_local]
        visitante_id = teams[equipo_visitante]

        df = get_h2h(local_id, visitante_id)
        if df.empty:
            st.warning("No se encontraron partidos históricos entre estos equipos.")
        else:
            st.success(f"Se encontraron {len(df)} partidos históricos:")
            st.dataframe(df)
