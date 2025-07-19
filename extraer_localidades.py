# extraer_localidades.py

import pickle

with open('model/modelo_xgb.pkl', 'rb') as f:
    _, columnas = pickle.load(f)

# Filtrar columnas que comienzan con "location_"
localidades = [col.replace('location_', '') for col in columnas if col.startswith('location_')]

# Guardar como archivo JS
with open('frontend/assets/js/localidades.js', 'w', encoding='utf-8') as f:
    f.write("const listaLocalidades = [\n")
    for loc in localidades:
        f.write(f'  "{loc}",\n')
    f.write("];\n")
