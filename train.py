import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Descarga y carga el dataset de calidad de vino."""
    df = pd.read_csv(url, sep=';')
    return df

def preprocess_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Limpia duplicados, binariza la variable objetivo y particiona los datos."""
    df_cleaned = df.drop_duplicates().copy()
    
    # Binarización: Calidad >= 7 es vino de alta calidad (1), menor es estándar/baja (0)
    df_cleaned['target'] = (df_cleaned['quality'] >= 7).astype(int)
    
    X = df_cleaned.drop(columns=['quality', 'target'])
    y = df_cleaned['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> LogisticRegression:
    """Entrena la regresión logística con los hiperparámetros dados."""
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series):
    """Evalúa el modelo y retorna métricas principales junto con la matriz de confusión."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average='weighted')),
        "f1_macro": float(f1_score(y_test, y_pred, average='macro'))
    }
    
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_pred

def save_confusion_matrix_artifact(cm: np.ndarray, output_path: str = "confusion_matrix.png") -> str:
    """Genera y guarda la figura de la matriz de confusión como artefacto."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Baja/Media', 'Alta'], 
        yticklabels=['Baja/Media', 'Alta']
    )
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path

def parse_args():
    """Expone hiperparámetros y semilla por línea de comandos con valores por defecto."""
    parser = argparse.ArgumentParser(description="Entrenamiento de baseline Wine Quality con MLflow")
    
    # Hiperparámetro 1: C (Inverso de regularización)
    parser.add_argument("--C", type=float, default=1.0, 
                        help="Inverso de la fuerza de regularización (default: 1.0)")
    # Hiperparámetro 2: solver
    parser.add_argument("--solver", type=str, default="liblinear", 
                        choices=["liblinear", "lbfgs", "saga"],
                        help="Algoritmo de optimización (default: liblinear)")
    # Hiperparámetro 3: max_iter
    parser.add_argument("--max_iter", type=int, default=500, 
                        help="Número máximo de iteraciones (default: 500)")
    # Semilla aleatoria
    parser.add_argument("--random_state", type=int, default=42, 
                        help="Semilla para reproducibilidad (default: 42)")
    # Nombre del experimento en MLflow
    parser.add_argument("--experiment_name", type=str, default="Wine_Quality_Experimentation",
                        help="Nombre del experimento en MLflow")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Fijación de semilla general
    np.random.seed(args.random_state)
    
    # Configurar el experimento en MLflow
    mlflow.set_experiment(args.experiment_name)
    
    params = {
        "C": args.C,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "random_state": args.random_state
    }
    
    # 1. Carga y preprocesamiento
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df, random_state=args.random_state)
    
    # 2. Tracking de la corrida en MLflow
    with mlflow.start_run():
        # Registrar hiperparámetros y metadatos
        mlflow.log_params(params)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("features_count", X_train.shape[1])
        
        # Entrenamiento
        model = train_model(X_train, y_train, params)
        
        # Evaluación
        metrics, cm, y_pred = evaluate_model(model, X_test, y_test)
        
        # Registrar métricas
        for metric_name, val in metrics.items():
            mlflow.log_metric(metric_name, val)
            
        # Generar y registrar artefacto (Matriz de confusión)
        artifact_path = save_confusion_matrix_artifact(cm)
        mlflow.log_artifact(artifact_path)
        if os.path.exists(artifact_path):
            os.remove(artifact_path)
            
        # Registrar modelo serializado
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # Resumen en consola
        print(f"[OK] Corrida guardada: C={args.C}, solver={args.solver}, max_iter={args.max_iter}, seed={args.random_state}")
        print(f"     Accuracy: {metrics['accuracy']:.4f} | F1 Weighted: {metrics['f1_weighted']:.4f} | F1 Macro: {metrics['f1_macro']:.4f}\n")

if __name__ == "__main__":
    main()