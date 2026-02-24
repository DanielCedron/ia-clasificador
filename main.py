import joblib
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

print("Despertando a la IA y cargando su cerebro...")
# 1. CARGAR EL CEREBRO DESDE EL DISCO DURO
# Ahora carga instantáneamente todo lo que aprendió en entrenar_ia.py
modelo_ia = joblib.load('cerebro_ia.pkl')

# 2. CREAR EL SERVIDOR WEB (API)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class MensajeInput(BaseModel):
    texto: str

@app.post("/analizar")
def analizar_mensaje(mensaje: MensajeInput):
    # La IA usa su cerebro guardado para predecir al instante
    prediccion = modelo_ia.predict([mensaje.texto])[0]
    return {"categoria": prediccion}



if __name__ == "__main__":
    import uvicorn
    # Render usa una variable de entorno llamada PORT
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)