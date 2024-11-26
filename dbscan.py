from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import os

def run_dbscan(input_file, output_dir="results/dbscan/"):
    os.makedirs(output_dir, exist_ok=True)

    # Leer el archivo CSV
    data = pd.read_csv(input_file)

    # Excluir columnas no numéricas
    numeric_data = data.select_dtypes(include=[float, int])

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=10)
    labels = dbscan.fit_predict(numeric_data)

    # Calcular Silhouette Score (ignora clusters con un solo elemento o ruido)
    try:
        silhouette = silhouette_score(numeric_data, labels)
    except ValueError as e:
        silhouette = -1  # Silhouette no es válido si hay ruido predominante

    # Guardar resultados
    data["cluster"] = labels
    data.to_csv(os.path.join(output_dir, "clusters.csv"), index=False)

    # Generar archivo metrics.txt
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Silhouette Score: {silhouette}\n")
        f.write(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}\n")
        f.write(f"Noise points: {list(labels).count(-1)}\n")

# Ejecución
if __name__ == "__main__":
    run_dbscan("result/VECTOR_train.csv")
