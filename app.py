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
        'p_over_0_5': p_over_0_5,
        'p_over_1_5': p_over_1_5,
        'p_over_2_5': p_over_2_5,
        'p_btts': p_btts,
        'p_h': prob_h_win,
        'p_d': prob_d,
        'p_a': prob_a_win,
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
    df['over_1_5'] = (df['total_goals'] >= 2).astype(int)
    df['over_2_5'] = (df['total_goals'] >= 3).astype(int)
    df['btts'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
    df['result'] = df.apply(lambda r: 'H' if r['home_goals']>r['away_goals'] else ('D' if r['home_goals']==r['away_goals'] else 'A'), axis=1)
    return df

# ------------------------
# TEAM STATS (fiable)
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
        s["media_gf"] = s["gf"]/s["jugados"] if s["jugados"]>0 else 0
        s["media_gc"] = s["gc"]/s["jugados"] if s["jugados"]>0 else 0
    return stats

# ------------------------
# FEATURE BUILDER for match prediction
# ------------------------
def build_match_feat(team_stats, home, away):
    s_h = team_stats.get(home, {})
    s_a = team_stats.get(away, {})
    hg = s_h.get("media_gf",0)
    hga = s_h.get("media_gc",0)
    ag = s_a.get("media_gf",0)
    aga = s_a.get("media_gc",0)
    lam_h = (hg/1.0)*(aga/1.0)*1.0
    lam_a = (ag/1.0)*(hga/1.0)*1.0
    return {
        'home_gf_avg': hg,'home_ga_avg':hga,
        'away_gf_avg': ag,'away_ga_avg':aga,
        'lam_home': lam_h,'lam_away': lam_a
    }

# ------------------------
# MODELING
# ------------------------
def train_ml_models(feat_df):
    feature_cols = ['home_gf_avg','home_ga_avg','away_gf_avg','away_ga_avg','lam_home','lam_away']
    X = feat_df[feature_cols].fillna(0)
    y_over1 = feat_df['over_1_5']
    y_over2 = feat_df['over_2_5']
    y_btts = feat_df['btts']
    y_res = feat_df['result']

    split = int(0.8*len(X)) if len(X)>10 else 0

    def train_bin(y):
        if split==0 or len(y.unique())<2: return None
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(X.iloc[:split,:],y.iloc[:split])
        return clf

    models={}
    models['over_1_5'] = train_bin(y_over1)
    models['over_2_5'] = train_bin(y_over2)
    models['btts'] = train_bin(y_btts)

    if split==0:
        models['1x2'] = None; models['le'] = None
    else:
        le = LabelEncoder()
        y_res_enc = le.fit_transform(y_res)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X.iloc[:split,:], y_res_enc[:split])
        models['1x2'] = clf; models['le'] = le
    return models, feature_cols

def predict_match_ml(models, feature_cols, match_feat):
    X_match = pd.DataFrame([match_feat])[feature_cols].fillna(0)
    ml_preds = {}
    for t in ['over_1_5','over_2_5','btts']:
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
    ensemble = {k: np.nanmean([poisson_probs.get(f'p_{k[-3:]}' if k!='btts' else 'p_btts',0), ml_preds.get(k,np.nan)]) for k in ['over_1_5','over_2_5','btts']}
    ensemble_1x2 = {k:(probs_1x2[k]+poisson_probs['p_h'] if k=='H' else (probs_1x2[k]+poisson_probs['p_d'] if k=='D' else probs_1x2[k]+poisson_probs['p_a']))/2 for k in probs_1x2}
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
for t in ['home_gf_avg','home_ga_avg','away_gf_avg','away_ga_avg','lam_home','lam_away']:
    if t not in feat_df.columns: feat_df[t]=0
models, feature_cols = train_ml_models(feat_df)
st.success("Modelos listos (si hay suficientes datos).")

# Selección de partido
equipos = sorted(set(df_raw['home_team']).union(df_raw['away_team']))
col1,col2 = st.columns(2)
with col1: equipo_local = st.selectbox("Equipo local", equipos, index=0)
with col2: equipo_visitante = st.selectbox("Equipo visitante", [e for e in equipos if e!=equipo_local], index=0)

if equipo_local and equipo_visitante:
    match_feat = build_match_feat(team_stats, equipo_local, equipo_visitante)
    st.markdown("### 🔧 Features estimadas del partido")
    st.write(match_feat)

    # Poisson
    poisson_probs = poisson_market_probs(match_feat['lam_home'], match_feat['lam_away'])
    st.markdown("### 🎲 Probabilidades (Poisson)")
    poisson_display = {k: round(v*100,1) for k,v in poisson_probs.items() if k != 'total_pmf'}
    st.write(poisson_display)
    st.markdown("Distribución total de goles (Poisson):")
    st.write(poisson_probs['total_pmf'].round(3).tolist())

    # ML
    ml_preds, probs_1x2 = predict_match_ml(models, feature_cols, match_feat)
    st.markdown("### 🤖 Predicciones ML (RandomForest)")
    st.write("ML probs (over1.5, over2.5, btts):", {k:(round(v*100,1) if not np.isnan(v) else "N/A") for k,v in ml_preds.items()})
    st.write("1X2 probs ML:", {k:round(v*100,1) for k,v in probs_1x2.items()})

    # Ensemble
    ensemble, ensemble_1x2 = ensemble_probs(poisson_probs, ml_preds, probs_1x2)
    st.markdown("### ✅ Probabilidades finales (ensamblado Poisson + ML)")
    st.write({k:round(v*100,1) for k,v in ensemble.items()})
    st.write("1X2 final:", {k:round(v*100,1) for k,v in ensemble_1x2.items()})

    st.markdown("### 📢 Recomendación de apuesta")
    recomendacion = generate_bet_recommendation_simple(ensemble_1x2, ensemble)
    st.write(recomendacion)

st.markdown("---")
st.write("Nota: este prototipo usa modelos simples. Mejora añadiendo más features y modelos (XGBoost, xG, tiros, descansos), además de calibración de probabilidades.")
