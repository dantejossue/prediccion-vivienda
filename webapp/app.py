from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar modelo y columnas
with open('../model/modelo_rf.pkl', 'rb') as f:
    best_model, columnas = pickle.load(f)

# Cargar preprocesador
with open('../model/preprocesador.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return "<h1>API de Predicción de Precios de Viviendas</h1><p>Usa /predict con método POST para estimar precios.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    metros = data['metros']
    habitaciones = data['habitaciones']
    banos = data['banos']
    localidad = data['localidad']

    # Crear DataFrame con formato original
    input_df = pd.DataFrame([{
        'total_sqft': metros,
        'bath': banos,
        'bhk(habitaciones)': habitaciones,
        'location': localidad.strip()
    }])

    # Aplicar preprocesamiento
    input_transformed = preprocessor.transform(input_df)

    # Predecir
    precio = best_model.predict(input_transformed)[0]

    return jsonify({'precio_estimado': float(round(precio, 2))})

if __name__ == '__main__':
    app.run(debug=True)
