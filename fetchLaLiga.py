import requests
import json

API_FOOTBALL_KEY = "c0dabdb4909f07174443de8fa7e175fa"
API_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}
fileName = "data/Historic.json"
def fetch_fixtures(league, season):
    url = f"{API_URL}/fixtures?league={league}&season={season}"
    r = requests.get(url, headers=HEADERS)
    data = r.json().get("response", [])
    matches = []
    for match in data:
        status = match["fixture"]["status"]["short"]
        if status == "FT":  # solo partidos finalizados
            matches.append({
                "season": season,
                "date": match["fixture"]["date"],
                "home_team": match["teams"]["home"]["name"],
                "away_team": match["teams"]["away"]["name"],
                "home_goals": match["goals"]["home"],
                "away_goals": match["goals"]["away"]
            })
    return matches

# Temporadas deseadas: 2017-2023 y 2025
seasons = list(range(2017, 2024)) + [2025]
all_matches = []

for season in seasons:
    print(f"Descargando temporada {season}...")
    try:
        all_matches.extend(fetch_fixtures(140, season))  # 140 = LaLiga
    except Exception as e:
        print(f"Error descargando temporada {season}: {e}")

# Guardamos todo en un JSON
with open(fileName, "w", encoding="utf-8") as f:
    json.dump(all_matches, f, ensure_ascii=False, indent=2)

print(f"Datos guardados en {fileName}")
