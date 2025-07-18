
# 🏘️ Estimador Inteligente de Precios de Viviendas en Bangalore

Este proyecto presenta un sistema de Machine Learning capaz de predecir el precio estimado de una vivienda en la ciudad de Bangalore, India, basado en características como área en metros cuadrados, número de habitaciones, baños y localidad. El modelo fue entrenado utilizando **RandomForestRegressor**, optimizado con validación cruzada, y desplegado a través de una API REST consumida por una interfaz web interactiva.

---

## 📌 Objetivos del Proyecto

- Aplicar técnicas supervisadas de aprendizaje automático.
- Automatizar el pipeline de entrenamiento, evaluación y despliegue.
- Desarrollar una interfaz que permita predecir precios desde el navegador.
- Integrar conocimientos de programación, ML y despliegue de sistemas.

---

## ⚙️ Estructura del Proyecto

```
prediccion_viviendas/
│
├── data/                   # Conjuntos de datos originales y transformados
│   ├── viviendas_bangalore_Datos_Limpios.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── model/                  # Modelos entrenados y scripts de ML
│   ├── entrenamiento.py
│   ├── evaluacion.py
│   ├── modelo_rf.pkl
│   ├── preprocesador.pkl
│   └── ...
│
├── webapp/                 # Backend con Flask
│   ├── app.py
│   └── extraer_localidades.py
│
├── frontend/               # Interfaz web
│   ├── index.html
│   └── assets/
│       ├── css/
│       ├── js/
│       └── img/
│
├── pipeline.py             # Script principal para correr el pipeline completo
├── requirements.txt        # Dependencias
└── README.md
```

---

## 🧠 Algoritmo utilizado

> **RandomForestRegressor**  
> Optimización mediante **GridSearchCV** (validación cruzada de 5 pliegues, 8 combinaciones evaluadas).

### 🔧 Mejores hiperparámetros encontrados:
```json
{
  "n_estimators": 150,
  "max_depth": 10,
  "min_samples_split": 2
}
```

---

## 📈 Métricas de Evaluación

Evaluación del modelo con el conjunto de prueba:

- MAE  : 15.30
- MSE  : 2123.90
- RMSE : 46.08
- R²   : 0.7486

---

## 🚀 Ejecución del Proyecto

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar y evaluar el modelo

```bash
python pipeline.py
```

### 3. Iniciar el backend Flask

```bash
cd webapp
python app.py
```

### 4. Abrir el frontend

Abre `frontend/index.html` en tu navegador. Ingresa los datos del inmueble y obtén una estimación inmediata.

---

## 🌐 Tecnologías Utilizadas

- Python
- Scikit-learn
- Pandas / NumPy
- Flask + CORS
- HTML, CSS, JS, Select2
- GridSearchCV para validación cruzada
- VS Code

---

## ✅ Buenas Prácticas del Pipeline

- Código modular y reutilizable (`entrenamiento.py`, `evaluacion.py`, etc.)
- Validación cruzada automatizada con GridSearchCV
- Separación de responsabilidades: `/model`, `/webapp`, `/frontend`
- Flujo reproducible desde cero mediante `pipeline.py`
- API REST clara, compatible con POST JSON
- Documentación completa en este `README.md`

---

## 👨‍💻 Autor

Desarrollado como parte del curso de Machine Learning.  
El proyecto fue implementado en Visual Studio Code y desplegado localmente.

