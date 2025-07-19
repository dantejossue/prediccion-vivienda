# model/entrenamiento.py

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def entrenar_modelo():
    # Cargar los datos transformados
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()

    # Definir el modelo base
    rf = RandomForestRegressor(random_state=42)

    # Grid de hiperparámetros
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    }

    # Validación cruzada
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)

    # Mejor modelo y columnas
    best_model = grid_search.best_estimator_
    columnas = X_train.columns.tolist()

    # Guardar modelo entrenado
    with open('model/modelo_rf.pkl', 'wb') as f:
        pickle.dump((best_model, columnas), f)

    print("✅ Modelo Random Forest entrenado y guardado con éxito.")
