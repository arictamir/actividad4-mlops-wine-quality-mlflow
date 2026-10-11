# Actividad: Código de Experimentación y Tracking con MLflow

Este repositorio contiene la instrumentación y modularización de un modelo base de Machine Learning (Regresión Logística sobre el dataset *Wine Quality*) para convertirlo en un pipeline de experimentación reproducible utilizando **MLflow Tracking** y la interfaz de línea de comandos (`argparse`).

---

## 📁 Estructura del Repositorio

```text
.
├── .gitignore          # Excluye archivos innecesarios
├── README.md           # Instrucciones de configuración y ejecución
├── notebook.ipynb      # Análisis Exploratorio de Datos (EDA)
├── requirements.txt    # Especificación de dependencias y versiones
└── train.py            # Script ejecutable de entrenamiento e instrumentación con MLflow
```

---

## 🛠️ Requisitos Previos e Instalación

1. **Clonar o descargar el repositorio:**
   ```bash
   git clone https://github.com/arictamir/actividad4-mlops-wine-quality-mlflow.git
   cd actividad4-mlops-wine-quality-mlflow
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   # En Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate

   # En Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🚀 Guía de Ejecución

El script `train.py` permite ejecutar el flujo completo de preprocesamiento, entrenamiento, cálculo de métricas y registro de artefactos en MLflow.

### 1. Ejecución con parámetros por defecto
Si se ejecuta sin argumentos, el script utiliza los valores predeterminados (`C=1.0`, `solver=liblinear`, `max_iter=500`, `random_state=42`):
```bash
python train.py
```

### 2. Parámetros de Línea de Comandos Disponibles
Puedes personalizar la ejecución pasando los siguientes argumentos:

| Argumento | Tipo | Default | Opciones / Descripción |
| :--- | :---: | :---: | :--- |
| `--C` | `float` | `1.0` | Inverso de la fuerza de regularización. |
| `--solver` | `str` | `liblinear` | Algoritmo de optimización (`liblinear`, `lbfgs`, `saga`). |
| `--max_iter` | `int` | `500` | Número máximo de iteraciones para convergencia. |
| `--random_state` | `int` | `42` | Semilla aleatoria para partición y entrenamiento (reproducibilidad). |
| `--experiment_name` | `str` | `Wine_Quality_Experimentation` | Nombre del experimento en MLflow. |

---

## 🧪 Ejecución de Corridas de Comparación

Para reproducir el experimento con al menos 5 configuraciones contrastables, ejecuta los siguientes comandos en tu terminal:

```bash
# Corrida 1: Configuración base por defecto
python train.py

# Corrida 2: Mayor regularización (C bajo)
python train.py --C 0.05 --solver liblinear --max_iter 500 --random_state 42

# Corrida 3: Menor regularización con solver lbfgs
python train.py --C 10.0 --solver lbfgs --max_iter 1000 --random_state 42

# Corrida 4: Regularización intermedia con solver saga
python train.py --C 0.5 --solver saga --max_iter 1000 --random_state 42

# Corrida 5: Prueba de sensibilidad a la semilla
python train.py --C 1.0 --solver liblinear --max_iter 500 --random_state 99
```

---

## 📊 Visualización de Resultados en MLflow UI

Para consultar el registro de métricas, hiperparámetros, artefactos (matriz de confusión) y comparar las corridas:

1. Levanta el servidor local de MLflow desde la raíz del proyecto:
   ```bash
   mlflow ui
   ```

2. Abre en tu navegador web:
   ```text
   http://127.0.0.1:5000
   ```

3. Dirígete al experimento **`Wine_Quality_Experimentation`**.
4. Selecciona las corridas registradas y haz clic en **Compare** para visualizar gráficos de dispersión, coordenadas paralelas y contrastar las métricas (`accuracy`, `f1_weighted`, `f1_macro`) y artefactos generados.