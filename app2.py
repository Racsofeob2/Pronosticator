# app.py
import requests, json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import math
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ------------------------
# CONFIG
# ------------------------
API_KEY = "e88d28a3765b4b69a32d756e1608b434"
API_URL = "https://api.football-data.org/v4/competitions/PD/matches"
HEADERS = {"X-Auth-Token": API_KEY}
DATA_FILE = Path("data/laliga.json")
DATA_FILE.parent.mkdir(exist_ok=True)

# ------------------------
# UTIL: Poisson
# ------------------------
def poisson_pmf(k, lam):
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except OverflowError:
        return 0.0

def poisson_distribution(lam, max_goals=10):
    pmf = np.array([poisson_pmf(k, lam) for k in range(max_goals + 1)])
    pmf = pmf / pmf.sum()
    return pmf

def total_goals_distribution(lam_h, lam_a, max_goals=10):
    pmf_h = poisson_distribution(lam_h, max_goals)
    pmf_a = poisson_distribution(lam_a, max_goals)
    total_len = 2 * max_goals + 1
    total = np.zeros(total_len)
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            total[i + j] += pmf_h[i] * pmf_a[j]
    return total

def poisson_market_probs(lam_h, lam_a, max_goals=10):
    total = total_goals_distribution(lam_h, lam_a, max_goals)
    p_over_0_5 = 1 - total[0]
    p_over_1_5 = 1 - total[:2].sum()
    p_over_2_5 = 1 - total[:3].sum()
    p_home0 = poisson_pmf(0, lam_h)
    p_away0 = poisson_pmf(0, lam_a)
    p_btts = 1 - p_home0 - p_away0 + p_home0 * p_away0

    pmf_h = poisson_distribution(lam_h, max_goals)
    pmf_a = poisson_distribution(lam_a, max_goals)
    prob_h_win = prob_d = prob_a_win = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pmf_h[i] * pmf_a[j]
            if i > j: prob_h_win += p
            elif i == j: prob_d += p
            else: prob_a_win += p
    return {
        'Over 0.5 goles': p_over_0_5,
        'Over 1.5 goles': p_over_1_5,
        'Over 2.5 goles': p_over_2_5,
        'Ambos marcan': p_btts,
        'Victoria local': prob_h_win,
        'Empate': prob_d,
        'Victoria visitante': prob_a_win,
        'total_pmf': total
    }

# ------------------------
# DATA FUNCTIONS
# ------------------------
def load_local_data():
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_local_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def update_data():
    r = requests.get(API_URL, headers=HEADERS)
    if r.status_code != 200:
        st.error(f"Error API: {r.status_code} - {r.text}")
        return load_local_data()
    matches = r.json().get("matches", [])
    new_records = []
    for m in matches:
        rec = {
            "id": m["id"],
            "utcDate": m["utcDate"],
            "matchday": m.get("matchday", None),
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
            "status": m["status"]
        }
        new_records.append(rec)
    local_data = load_local_data()
    all_data = {str(m["id"]): m for m in (local_data + new_records)}
    all_data = list(all_data.values())
    save_local_data(all_data)
    return all_data

def normalize_df(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if 'utcDate' in df.columns:
        df['utcDate'] = pd.to_datetime(df['utcDate'])
    return df

def prepare_outcomes(df):
    df = df.copy()
    df['total_goals'] = df['home_goals'] + df['away_goals']
    df['Over 1.5 goles'] = (df['total_goals'] >= 2).astype(int)
    df['Over 2.5 goles'] = (df['total_goals'] >= 3).astype(int)
    df['Ambos marcan'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
    df['result'] = df.apply(lambda r: 'H' if r['home_goals']>r['away_goals'] else ('D' if r['home_goals']==r['away_goals'] else 'A'), axis=1)
    return df

# ------------------------
# TEAM STATS
# ------------------------
def compute_team_stats(df):
    stats = defaultdict(lambda: {"jugados":0,"gf":0,"gc":0,"w":0,"d":0,"l":0})
    for _, row in df.iterrows():
        if row['status'] != "FINISHED": continue
        home = row['home_team']
        away = row['away_team']
        hg = row['home_goals']; ag = row['away_goals']
        stats[home]["jugados"] +=1; stats[home]["gf"] += hg; stats[home]["gc"] += ag
        stats[away]["jugados"] +=1; stats[away]["gf"] += ag; stats[away]["gc"] += hg
        if hg>ag: stats[home]["w"]+=1; stats[away]["l"]+=1
        elif hg==ag: stats[home]["d"]+=1; stats[away]["d"]+=1
        else: stats[home]["l"]+=1; stats[away]["w"]+=1
    for team in stats:
        s = stats[team]
        s["Media goles local"] = s["gf"]/s["jugados"] if s["jugados"]>0 else 0
        s["Media goles encajados local"] = s["gc"]/s["jugados"] if s["jugados"]>0 else 0
    return stats

# ------------------------
# FEATURE BUILDER
# ------------------------
def build_match_feat(team_stats, home, away):
    s_h = team_stats.get(home, {})
    s_a = team_stats.get(away, {})
    return {
        "Media goles local": s_h.get("Media goles local",0),
        "Media goles encajados local": s_h.get("Media goles encajados local",0),
        "Media goles visitante": s_a.get("Media goles local",0),
        "Media goles encajados visitante": s_a.get("Media goles encajados local",0),
        "Lambda local": s_h.get("Media goles local",0) * s_a.get("Media goles encajados local",0),
        "Lambda visitante": s_a.get("Media goles local",0) * s_h.get("Media goles encajados local",0)
    }

# ------------------------
# MODELING
# ------------------------
def train_ml_models(feat_df):
    feature_cols = ['Media goles local','Media goles encajados local',
                    'Media goles visitante','Media goles encajados visitante',
                    'Lambda local','Lambda visitante']
    X = feat_df[feature_cols].fillna(0)
    targets = ['Over 1.5 goles','Over 2.5 goles','Ambos marcan','result']

    if len(X) < 10:
        st.warning("No hay suficientes datos para entrenar los modelos ML.")
        return {}, feature_cols

    models = {}
    for t in targets[:-1]:  # binarios
        y = feat_df[t]
        if len(y.unique()) < 2:
            st.warning(f"No hay suficientes clases para entrenar {t}.")
            models[t] = None
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        st.info(f"Modelo {t} entrenado. Accuracy test: {score:.2f}")
        models[t] = clf

    # Modelo 1X2
    y_res = feat_df['result']
    if len(y_res.unique()) < 2:
        st.warning("No hay suficientes clases para entrenar 1X2.")
        models['1x2'] = None
        models['le'] = None
    else:
        le = LabelEncoder()
        y_res_enc = le.fit_transform(y_res)
        X_train, X_test, y_train, y_test = train_test_split(X, y_res_enc, test_size=0.2, random_state=42)
        clf3 = RandomForestClassifier(n_estimators=200, random_state=42)
        clf3.fit(X_train, y_train)
        score = clf3.score(X_test, y_test)
        st.info(f"Modelo 1X2 entrenado. Accuracy test: {score:.2f}")
        models['1x2'] = clf3
        models['le'] = le

    return models, feature_cols

def predict_match_ml(models, feature_cols, match_feat):
    X_match = pd.DataFrame([match_feat])[feature_cols].fillna(0)
    ml_preds = {}
    for t in ['Over 1.5 goles','Over 2.5 goles','Ambos marcan']:
        mdl = models.get(t)
        if mdl is None: ml_preds[t] = np.nan
        else:
            try: ml_preds[t] = float(mdl.predict_proba(X_match)[:,1][0])
            except: ml_preds[t] = float(mdl.predict(X_match)[0])

    if models.get('1x2') is not None:
        clf3 = models['1x2']; le = models['le']
        p_1x2 = clf3.predict_proba(X_match)[0]
        probs_1x2 = {le.inverse_transform([i])[0]: float(p) for i,p in enumerate(p_1x2)}
    else:
        probs_1x2 = {'H':0.33,'D':0.33,'A':0.33}

    return ml_preds, probs_1x2

def ensemble_probs(poisson_probs, ml_preds, probs_1x2):
    ensemble = {k: np.nanmean([poisson_probs.get(k,0), ml_preds.get(k,np.nan)]) for k in ['Over 1.5 goles','Over 2.5 goles','Ambos marcan']}
    ensemble_1x2 = {k:(probs_1x2[k]+poisson_probs['Victoria local'] if k=='H' else (probs_1x2[k]+poisson_probs['Empate'] if k=='D' else probs_1x2[k]+poisson_probs['Victoria visitante']))/2 for k in probs_1x2}
    return ensemble, ensemble_1x2

def generate_bet_recommendation_simple(ensemble_1x2, ensemble):
    best = max(ensemble_1x2, key=ensemble_1x2.get)
    if ensemble_1x2[best]<0.5: return "No se recomienda apostar según los datos actuales."
    return f"Pues analizando los datos, deberías apostar a: {'LOCAL (H)' if best=='H' else 'EMPATE (D)' if best=='D' else 'VISITANTE (A)'}."

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="LaLiga AI Predictor (local)", layout="wide")
st.title("⚽ LaLiga — Análisis + Pronóstico (Poisson + ML)")

if st.button("Actualizar datos desde API"):
    raw = update_data(); st.success("Datos actualizados desde API")
else: raw = load_local_data()
if not raw: st.warning("No hay datos. Pulsa 'Actualizar datos desde API'."); st.stop()

df_raw = normalize_df(pd.DataFrame(raw))
df_prepared = prepare_outcomes(df_raw)

# Estadísticas fiables de equipos
team_stats = compute_team_stats(df_raw)
st.markdown("### 📊 Estadísticas de cada equipo")
df_team_stats = pd.DataFrame(team_stats).T.sort_values(by="jugados",ascending=False)
st.dataframe(df_team_stats)

# Entrenamos modelos
st.info("Entrenando modelos ML con los datos locales (puede tardar unos segundos)...")
feat_df = df_prepared.copy()
for t in ['Media goles local','Media goles encajados local','Media goles visitante','Media goles encajados visitante','Lambda local','Lambda visitante']:
    if t not in feat_df.columns: feat_df[t]=0
models, feature_cols = train_ml_models(feat_df)
st.success("Modelos listos (si hay suficientes datos).")

# ------------------------
# Selección de próximo partido
# ------------------------
upcoming_matches = df_raw[df_raw['status'] != "FINISHED"].sort_values('utcDate')
if upcoming_matches.empty:
    st.warning("No hay próximos partidos disponibles en la API.")
    st.stop()

match_options = [f"{row['home_team']} vs {row['away_team']} ({row['utcDate'].strftime('%d-%m %H:%M')})"
                 for _, row in upcoming_matches.iterrows()]
selected_match = st.selectbox("Selecciona un próximo partido:", match_options, index=0)
match_row = upcoming_matches.iloc[match_options.index(selected_match)]

equipo_local = match_row['home_team']
equipo_visitante = match_row['away_team']

# ------------------------
# Predicción del partido
# ------------------------
match_feat = build_match_feat(team_stats, equipo_local, equipo_visitante)
st.markdown("### 🔧 Features estimadas del partido")
st.write(match_feat)

# Poisson
poisson_probs = poisson_market_probs(match_feat['Lambda local'], match_feat['Lambda visitante'])
st.markdown("### 🎲 Probabilidades (Poisson)")
poisson_display = {k:v for k,v in poisson_probs.items() if k != 'total_pmf'}
st.write(poisson_display)
st.markdown("Distribución total de goles (Poisson):")
st.write(poisson_probs['total_pmf'].round(3).tolist())

# ML
ml_preds, probs_1x2 = predict_match_ml(models, feature_cols, match_feat)
st.markdown("### 🤖 Predicciones ML (RandomForest)")
st.write("ML probs:", ml_preds)
st.write("1X2 probs ML:", probs_1x2)

# Ensemble
ensemble, ensemble_1x2 = ensemble_probs(poisson_probs, ml_preds, probs_1x2)
st.markdown("### ✅ Probabilidades finales (ensamblado Poisson + ML)")
st.write(ensemble)
st.write("1X2 final:", ensemble_1x2)

st.markdown("### 📢 Recomendación de apuesta")
recomendacion = generate_bet_recommendation_simple(ensemble_1x2, ensemble)
st.write(recomendacion)

st.markdown("---")
st.write("Nota: este prototipo usa modelos simples. Mejora añadiendo más features y calibración de probabilidades.")
