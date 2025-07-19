# model/evaluacion.py

import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluar_modelo():
    # Cargar modelo entrenado
    with open('model/modelo_rf.pkl', 'rb') as f:
        modelo, columnas = pickle.load(f)

    # Cargar datos de prueba
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()

    # Predicción
    y_pred = modelo.predict(X_test)

    # Evaluación
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("📊 Evaluación del modelo Random Forest:")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
