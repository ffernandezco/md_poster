import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os


def evaluate_model(y_true, y_pred, labels, output_dir):
    """
    Evalúa el modelo usando métricas estándar y genera un reporte de clasificación.
    """
    # Cálculo de métricas con manejo de cero en división
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

    # Imprimir métricas de resumen
    print("Summary Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print()

    # Reporte de clasificación
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels))

    # Guardar resultados en archivo
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = os.path.join(output_dir, "classification_report.txt")
    with open(result_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(classification_report(y_true, y_pred, labels=labels)))

    print(f"Results saved to {result_file}")


def main():
    # Cargar el archivo con los resultados del clustering
    df_cluster = pd.read_csv("result/CLUSTER_result.csv")

    # Cargar el archivo con las clases reales
    df_classes = pd.read_csv("result/Classes.csv", sep="|")

    # Fusionar ambos DataFrames por la columna 'ID'
    df_merged = pd.merge(df_cluster, df_classes, on="ID", how="inner")

    # Las etiquetas reales están en la columna 'PCM'
    y_true = df_merged["RS"].tolist()

    # Las etiquetas predichas por DBSCAN están en la columna 'Cluster'
    y_pred = df_merged["Cluster"].tolist()

    # Imprimir la distribución de clases para detectar problemas
    print("Distribución de clases reales (y_true):")
    print(pd.Series(y_true).value_counts())
    print("Distribución de clases predichas (y_pred):")
    print(pd.Series(y_pred).value_counts())

    # Etiquetas posibles (puedes ajustarlas dependiendo de tu caso)
    labels = list(set(y_true))  # Las etiquetas deben ser únicas en y_true

    # Directorio para guardar los resultados
    output_directory = "results"

    # Evaluar el modelo
    evaluate_model(y_true, y_pred, labels, output_dir=output_directory)


if __name__ == "__main__":
    main()
