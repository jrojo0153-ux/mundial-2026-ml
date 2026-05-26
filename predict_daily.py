import pandas as pd
import joblib
import requests
from datetime import datetime

# Cargar modelo y codificador
modelo = joblib.load('models/modelo_mundial.pkl')
le = joblib.load('models/codificador_equipos.pkl')

def predecir_resultado(local, visitante):
    try:
        # Si el equipo no existe en el codificador, lo manejamos
        if local not in le.classes_ or visitante not in le.classes_:
            return "Duelo Nuevo", "50%"
        
        id_l = le.transform([local])[0]
        id_v = le.transform([visitante])[0]
        
        prob = modelo.predict_proba([[id_l, id_v]])[0]
        
        # Lógica de decisión
        if prob[0] > prob[2] and prob[0] > prob[1]:
            return f"Gana {local}", f"{prob[0]*100:.1f}%"
        elif prob[2] > prob[0] and prob[2] > prob[1]:
            return f"Gana {visitante}", f"{prob[2]*100:.1f}%"
        else:
            return "Empate", f"{prob[1]*100:.1f}%"
    except:
        return "Análisis pendiente", "N/A"

# Obtener partidos de ESPN
url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
data = requests.get(url).json()

partidos_lista = []
for evento in data.get('events', []):
    comp = evento['competitions'][0]
    loc = comp['competitors'][0]['team']['displayName']
    vis = comp['competitors'][1]['team']['displayName']
    
    pred, conf = predecir_resultado(loc, vis)
    
    partidos_lista.append({
        'Partido': f"{loc} vs {vis}",
        'Predicción': pred,
        'Confianza': conf
    })

# Crear README
fecha = datetime.now().strftime("%Y-%m-%d")
df = pd.DataFrame(partidos_lista)
tabla = df.to_markdown(index=False) if not df.empty else "No hay partidos para hoy."

with open("README.md", "w") as f:
    f.write(f"# 🏆 Predicciones Mundial 2026\n\n### 📅 Hoy: {fecha}\n\n{tabla}\n\n---\n_Actualizado automáticamente_")
