import preprocess
import vectorize
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from statistics import mean, stdev


def run_script(script_name, args):
    cmd = ["python", script_name] + args
    subprocess.run(cmd, check=True)


def collect_metrics(result_dirs, iterations):
    metrics = []
    for method, path in result_dirs.items():
        for i in range(iterations):
            iteration_path = os.path.join(path, f"iteration_{i + 1}")
            metrics_path = os.path.join(iteration_path, "metrics.txt")
            try:
                with open(metrics_path, "r") as f:
                    lines = f.readlines()

                    # Extraer la accuracy de la primera línea
                    accuracy = float(lines[0].split(":")[1].strip())

                    # Buscar el Precision, F1-score y Recall de la clase 1
                    precision = None
                    f1_score = None
                    recall = None
                    for line in lines:
                        if line.strip().startswith("1"):  # Línea que comienza con la clase 1
                            precision = float(line.split()[1])  # Precision está en la columna 2
                            recall = float(line.split()[2])  # Recall está en la columna 3
                            f1_score = float(line.split()[3])  # F1-score está en la columna 4
                            break

                    if precision is None or f1_score is None or recall is None:
                        raise ValueError(f"Precision, F1-score o Recall de la clase 1 no encontrado en {metrics_path}")

                    metrics.append(
                        {"method": method, "accuracy": accuracy, "precision": precision, "f1_score": f1_score,
                         "recall": recall})
            except (IndexError, ValueError, FileNotFoundError) as e:
                print(f"Error procesando {metrics_path}: {e}")
                continue

    return pd.DataFrame(metrics)


def plot_class_distribution(train_file):
    df = pd.read_csv(train_file, low_memory=False)
    rs_counts = df['RS'].value_counts()

    plt.figure(figsize=(10, 6))
    rs_counts.plot(kind='bar')
    plt.title('Class Distribution (Train)')
    plt.xlabel('RS')
    plt.ylabel('Número de muestras')
    plt.xticks(rotation=0)

    for i, v in enumerate(rs_counts):
        plt.text(i, v, str(v), ha='center', va='bottom')

    plt.text(0.5, -0.15, f'Total: {len(df)}',
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes)

    plt.tight_layout()
    os.makedirs('results/comparison', exist_ok=True)
    plt.savefig('results/comparison/class_distribution.png')
    plt.close()


def plot_f1_recall_precision_comparison(metrics_df):
    summary_df = metrics_df.groupby("method").agg(
        mean_f1_score=("f1_score", mean),
        std_f1_score=("f1_score", stdev),
        mean_recall=("recall", mean),
        std_recall=("recall", stdev),
        mean_precision=("precision", mean),
        std_precision=("precision", stdev),
        mean_accuracy=("accuracy", mean),
        std_accuracy=("accuracy", stdev)
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Graficar las barras del F1-score
    sns.barplot(x="method", y="mean_f1_score", data=summary_df, palette="viridis", ax=ax1, label="F1-Score")

    # Añadir barras de error manualmente para evitar el error de forma
    for i in range(len(summary_df)):
        ax1.errorbar(i, summary_df["mean_f1_score"][i], yerr=summary_df["std_f1_score"][i], fmt='none', c='blue',
                     capsize=5)

    ax1.set_ylabel("F1-Score", color="blue")
    ax1.set_xlabel("Método")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Crear una segunda escala para el recall, precision y accuracy
    ax2 = ax1.twinx()

    # Añadir línea para el recall
    ax2.plot(summary_df["method"], summary_df["mean_recall"], marker="o", color="red", label="Recall", linewidth=2)

    # Añadir barras de error manualmente para evitar el error de forma
    for i in range(len(summary_df)):
        ax2.errorbar(i, summary_df["mean_recall"][i], yerr=summary_df["std_recall"][i], fmt='none', c='red', capsize=5)

    # Añadir línea para el precision
    ax2.plot(summary_df["method"], summary_df["mean_precision"], marker="o", color="orange", label="Precision",
             linewidth=2)

    # Añadir barras de error manualmente para evitar el error de forma
    for i in range(len(summary_df)):
        ax2.errorbar(i, summary_df["mean_precision"][i], yerr=summary_df["std_precision"][i], fmt='none', c='orange',
                     capsize=5)

    # Añadir línea para el accuracy
    ax2.plot(summary_df["method"], summary_df["mean_accuracy"], marker="o", color="green", label="Accuracy",
             linewidth=2)

    # Añadir barras de error manualmente para evitar el error de forma
    for i in range(len(summary_df)):
        ax2.errorbar(i, summary_df["mean_accuracy"][i], yerr=summary_df["std_accuracy"][i], fmt='none', c='green',
                     capsize=5)

    ax2.set_ylabel("Recall / Precision / Accuracy", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Personalizar el gráfico
    plt.title("F1-Score, Recall, Precision y Accuracy Comparison (Clase 1 - Minoritaria)")
    plt.xticks(rotation=45)

    # Agregar leyenda
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()

    os.makedirs("results/comparison", exist_ok=True)
    plt.savefig("results/comparison/f1_recall_precision_comparison.png")
    plt.close()


def plot_f1_boxplot(metrics_df):
    plt.figure(figsize=(10, 6))

    sns.boxplot(x="method", y="f1_score", data=metrics_df, palette="viridis")

    plt.title("Boxplot of F1-Score by Method")
    plt.xlabel("Method")
    plt.ylabel("F1-Score")

    plt.tight_layout()

    os.makedirs("results/comparison", exist_ok=True)
    plt.savefig("results/comparison/f1_boxplot.png")
    plt.close()

def plot_doc_length_boxplot(input_file):
    """
    Genera un box-plot para la longitud de los documentos en el archivo original.
    """
    df = pd.read_csv(input_file, low_memory=False)

    # Comprobar si existe la columna 'TXT'
    if 'TXT' not in df.columns:
        raise ValueError("La columna 'TXT' no existe en el archivo de entrada.")

    # Calcular la longitud de los documentos
    df['doc_length'] = df['TXT'].str.split().apply(len)

    plt.figure(figsize=(10, 6))
    sns.boxplot(y=df['doc_length'], palette="Set2")
    plt.title('Longitud de los textos')
    plt.ylabel('Número de palabras')

    plt.tight_layout()
    os.makedirs('results/comparison', exist_ok=True)
    plt.savefig('results/comparison/doc_length_boxplot.png')
    plt.close()


# Crear carpetas necesarias
if not os.path.exists("data/"):
    os.makedirs("data/")
if not os.path.exists("result/"):
    os.makedirs("result/")
if not os.path.exists("model/"):
    os.makedirs("model/")

# PREPROCESADO
input_file = 'data/DataI_MD.csv'
output_file = 'data/DataI_MD_POST.csv'
preprocess.preprocess(input_file, output_file)

# VECTORIZACIÓN
vector_input = output_file
vector_output = 'result/DataI_MD_VECTOR.csv'

vector_data = pd.read_csv(vector_output, delimiter="|")
df_classes = vector_data[["ID", "PCM", "RS", "PU"]]
df_classes.to_csv('result/Classes.csv', sep="|", index=False)

vector_data_cleaned = vector_data.drop(columns=["TXT"], errors='ignore')
cleaned_output = 'result/DataI_MD_VECTOR_CLEAN.csv'
vector_data_cleaned.to_csv(cleaned_output, sep="|", index=False)

train_file = 'result/VECTOR_train.csv'
test_file = 'result/VECTOR_test.csv'
preprocess.divide_csv(cleaned_output, train_file, test_file, 0.85, delimiter="|")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)

    # Gráfico de distribución de clases
    plot_class_distribution(train_file)

    # Box-plot de longitudes de documentos
    plot_doc_length_boxplot(input_file)

    iterations = 5  # Número de iteraciones para cada modelo

    for i in range(iterations):
        print(f"Iteración {i + 1}...")

        print("Ejecutando reducción dimensional PCA...")
        run_script("rf-pca.py", [f"iteration_{i + 1}"])

        print("Ejecutando reducción dimensional UMAP...")
        run_script("rf-umap.py", [f"iteration_{i + 1}"])

        print("Ejecutando Random Forest sin reducción de dimensionalidad...")
        run_script("rf.py", [f"iteration_{i + 1}"])

        # Copiar resultados a una ruta temporal para comparaciones posteriores
        for method in ["pca", "umap", "rf"]:
            src_dir = f"results/{method}/"
            dest_dir = f"results/{method}/iteration_{i + 1}/"
            shutil.copytree(src_dir, dest_dir)

    print("Comparando métricas...")
    result_dirs = {
        "Base": "results/rf/",
        "PCA": "results/pca/",
        "UMAP": "results/umap/"
    }

    metrics_df = collect_metrics(result_dirs, iterations)

    metrics_df.to_csv("results/comparison/metrics_summary.csv", index=False)

    print("Generando gráficos...")
    plot_f1_recall_precision_comparison(metrics_df)
    plot_f1_boxplot(metrics_df)

    print("Proceso completado. Gráficos guardados en results/comparison/")