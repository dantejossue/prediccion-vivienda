# pipeline.py

from model.entrenamiento import entrenar_modelo
from model.evaluacion import evaluar_modelo

if __name__ == "__main__":
    print("🚀 Ejecutando pipeline completo de entrenamiento y evaluación...")
    entrenar_modelo()
    evaluar_modelo()

