import pandas as pd
import joblib
import requests
from datetime import datetime

# Cargar Modelo V2 y Diccionario de Rankings
modelo = joblib.load('models/modelo_mundial_v2.pkl')
ranking_dict = joblib.load('models/ranking_dict.pkl')

def get_rank(pais):
    return ranking_dict.get(pais, 100)

def predecir_v2(local, visitante):
    r_l = get_rank(local)
    r_v = get_rank(visitante)
    diff = r_l - r_v
    
    # La IA ahora analiza la diferencia de nivel
    prob = modelo.predict_proba([[r_l, r_v, diff]])[0]
    
    opciones = [f"Gana {local}", "Empate", f"Gana {visitante}"]
    idx = prob.argmax()
    
    return opciones[idx], f"{prob[idx]*100:.1f}%"

# Obtener partidos de hoy de ESPN
try:
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    data = requests.get(url).json()
    
    partidos_lista = []
    for evento in data.get('events', []):
        comp = evento['competitions'][0]
        loc = comp['competitors'][0]['team']['displayName']
        vis = comp['competitors'][1]['team']['displayName']
        
        pred, conf = predecir_v2(loc, vis)
        partidos_lista.append({'Partido': f"{loc} vs {vis}", 'Predicción': pred, 'Confianza': conf})

    df = pd.DataFrame(partidos_lista)
    tabla = df.to_markdown(index=False) if not df.empty else "No hay partidos hoy."
except:
    tabla = "Error al obtener partidos o no hay juegos hoy."

# Escribir README
fecha = datetime.now().strftime("%Y-%m-%d")
with open("README.md", "w") as f:
    f.write(f"# 🏆 IA Mundial 2026 (V2 - High Accuracy)\n\n")
    f.write(f"### 📅 Predicciones para hoy: {fecha}\n\n{tabla}\n\n")
    f.write("---\n_IA mejorada con Ranking FIFA y Random Forest_")
