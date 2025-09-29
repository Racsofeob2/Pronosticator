# historic_viewer.py
import streamlit as st
import json
import pandas as pd

# Cargamos los datos históricos
DATA_FILE = "data/Historic.json"

with open(DATA_FILE, "r", encoding="utf-8") as f:
    all_matches = json.load(f)

# Convertimos a DataFrame
df = pd.DataFrame(all_matches)
df['season'] = df['season'].astype(int)

st.set_page_config(page_title="LaLiga Histórico", layout="wide")
st.title("📅 Histórico de partidos de LaLiga (2017-2023 y 2025)")

# Selección de temporada
seasons = sorted(df['season'].unique())
selected_season = st.selectbox("Selecciona la temporada:", seasons)

# Filtramos por temporada
df_season = df[df['season'] == selected_season].sort_values('date')

st.markdown(f"### Partidos de la temporada {selected_season}")
st.dataframe(df_season[['date', 'home_team', 'away_team', 'home_goals', 'away_goals']])

# -------------------------------
# Tabla de clasificación
# -------------------------------
def compute_standings(df_matches):
    teams = pd.unique(df_matches[['home_team','away_team']].values.ravel('K'))
    table = {team: {'PJ':0, 'G':0, 'E':0, 'P':0, 'GF':0, 'GC':0, 'DG':0, 'PTS':0} for team in teams}
    
    for _, row in df_matches.iterrows():
        h, a = row['home_team'], row['away_team']
        hg, ag = row['home_goals'], row['away_goals']
        if hg is None or ag is None:
            continue
        
        # Partidos jugados
        table[h]['PJ'] += 1
        table[a]['PJ'] += 1
        
        # Goles
        table[h]['GF'] += hg
        table[h]['GC'] += ag
        table[a]['GF'] += ag
        table[a]['GC'] += hg
        
        # Diferencia de goles
        table[h]['DG'] = table[h]['GF'] - table[h]['GC']
        table[a]['DG'] = table[a]['GF'] - table[a]['GC']
        
        # Resultado y puntos
        if hg > ag:
            table[h]['G'] += 1
            table[a]['P'] += 1
            table[h]['PTS'] += 3
        elif hg < ag:
            table[a]['G'] += 1
            table[h]['P'] += 1
            table[a]['PTS'] += 3
        else:
            table[h]['E'] += 1
            table[a]['E'] += 1
            table[h]['PTS'] += 1
            table[a]['PTS'] += 1
    
    standings_df = pd.DataFrame(table).T
    standings_df = standings_df.sort_values(['PTS','DG','GF'], ascending=[False,False,False])
    return standings_df

st.markdown(f"### 🏆 Tabla de clasificación temporada {selected_season}")
df_standings = compute_standings(df_season)
st.dataframe(df_standings)
