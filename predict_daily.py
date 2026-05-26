import pandas as pd
import joblib
import requests
from datetime import datetime

# 1. Cargar el modelo que subimos al repo
modelo = joblib.load('models/modelo_mundial.pkl')
le = joblib.load('models/codificador_equipos.pkl')

def obtener_partidos_dia():
    # API de ESPN para los partidos de hoy
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    res = requests.get(url).json()
    
    partidos_hoy = []
    for evento in res.get('events', []):
        comp = evento['competitions'][0]
        partidos_hoy.append({
            'local': comp['competitors'][0]['team']['displayName'],
            'visitante': comp['competitors'][1]['team']['displayName'],
            'hora': evento['date']
        })
    return partidos_hoy

def predecir():
    partidos = obtener_partidos_dia()
    if not partidos:
        return "### 📭 No hay partidos programados para hoy."

    resultados = []
    for p in partidos:
        try:
            # Codificar equipos
            id_l = le.transform([p['local']])[0]
            id_v = le.transform([p['visitante']])[0]
            
            # Predecir
            prob = modelo.predict_proba([[id_l, id_v]])[0]
            ganador = p['local'] if prob[0] > prob[2] else p['visitante']
            conf = max(prob) * 100
            
            resultados.append({
                'Partido': f"{p['local']} vs {p['visitante']}",
                'Predicción': f"Gana {ganador}",
                'Confianza': f"{conf:.1f}%"
            })
        except:
            resultados.append({
                'Partido': f"{p['local']} vs {p['visitante']}",
                'Predicción': "Datos insuficientes",
                'Confianza': "N/A"
            })
    
    df = pd.DataFrame(resultados)
    return df.to_markdown(index=False)

# Generar el nuevo README
fecha = datetime.now().strftime("%Y-%m-%d")
tabla = predecir()

nuevo_readme = f"""
# 🏆 Predicciones Diarias - Mundial 2026
> Actualizado automáticamente por la IA el: {fecha}

### 📅 Partidos de Hoy y Predicciones
{tabla}

---
_Este sistema se ejecuta automáticamente cada 24 horas usando GitHub Actions y el modelo entrenado en Colab._
"""

with open("README.md", "w") as f:
    f.write(nuevo_readme)
