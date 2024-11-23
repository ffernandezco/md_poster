import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import os


def train_and_evaluate(train_file, output_file, eps=0.5, min_samples=5, n_components=50):
    # Cargar datos
    data = pd.read_csv(train_file)

    # Separar ID y embeddings
    ids = data['ID']
    embeddings = data.drop(columns=['ID'])

    # Aplicar PCA para reducir la dimensionalidad
    pca = PCA(n_components=n_components)  # Ajustar el número de componentes según sea necesario
    embeddings_reduced = pca.fit_transform(embeddings)

    # Entrenar modelo DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(embeddings_reduced)

    # Guardar resultados del clustering
    result_df = data.copy()
    result_df['Cluster'] = cluster_labels
    result_df.to_csv(output_file, index=False)

    # Evaluar: Calcular class-to-cluster accuracy si las etiquetas reales existen
    # (PCM, RS, PU) no están en este archivo, así que simularemos el proceso para ahora.
    print("Clustering completado. Resultados guardados en:", output_file)

    # Si deseas usar métricas reales, adapta las etiquetas aquí
    # Ejemplo para generar una matriz de confusión (requiere etiquetas verdaderas)
    # y calcular class-to-cluster accuracy.

    return cluster_labels


if __name__ == "__main__":
    # Rutas de entrada/salida
    train_file = 'result/VECTOR_train.csv'  # Datos de entrenamiento
    output_file = 'result/CLUSTER_result.csv'  # Resultados del clustering

    # Parámetros del modelo DBSCAN
    eps = 0.3  # Distancia máxima para considerar vecinos
    min_samples = 7  # Mínimo de muestras en el vecindario
    n_components = 20  # Número de componentes principales a usar en PCA

    # Crear carpeta para resultados si no existe
    os.makedirs('result/', exist_ok=True)

    # Entrenar y evaluar
    train_and_evaluate(train_file, output_file, eps, min_samples, n_components)
