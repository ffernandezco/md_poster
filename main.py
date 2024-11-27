import preprocess
import vectorize
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_script(script_name, args):
    cmd = ["python", script_name] + args
    subprocess.run(cmd, check=True)


def collect_metrics(result_dirs):
    metrics = []
    for method, path in result_dirs.items():
        metrics_path = os.path.join(path, "metrics.txt")
        try:
            with open(metrics_path, "r") as f:
                lines = f.readlines()

                # Extraer la accuracy de la primera línea
                accuracy = float(lines[0].split(":")[1].strip())

                # Buscar el F1-score y Recall de la clase 1
                f1_score = None
                recall = None
                for line in lines:
                    if line.strip().startswith("1"):  # Línea que comienza con la clase 1
                        f1_score = float(line.split()[3])  # F1-score está en la columna 4
                        recall = float(line.split()[2])  # Recall está en la columna 3
                        break

                if f1_score is None or recall is None:
                    raise ValueError(f"F1-score o Recall de la clase 1 no encontrado en {metrics_path}")

                metrics.append({"method": method, "accuracy": accuracy, "f1_score": f1_score, "recall": recall})
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


def plot_f1_recall_comparison(metrics_df):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Graficar las barras del F1-score
    sns.barplot(x="method", y="f1_score", data=metrics_df, errorbar='sd', palette="viridis", ax=ax1, label="F1-Score")
    ax1.set_ylabel("F1-Score", color="blue")
    ax1.set_xlabel("Método")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Crear una segunda escala para el recall
    ax2 = ax1.twinx()
    ax2.plot(metrics_df["method"], metrics_df["recall"], marker="o", color="red", label="Recall", linewidth=2)
    ax2.set_ylabel("Recall", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Personalizar el gráfico
    plt.title("F1-Score y Recall Comparison (Clase 1 - Minoritaria)")
    plt.xticks(rotation=45)

    # Agregar leyenda
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()

    os.makedirs("results/comparison", exist_ok=True)
    plt.savefig("results/comparison/f1_recall_comparison.png")
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

    plot_class_distribution(train_file)

    print("Ejecutando reducción dimensional t-SNE...")
    run_script("rf-tsne.py", [])

    print("Ejecutando reducción dimensional UMAP...")
    run_script("rf-umap.py", [])

    print("Ejecutando clustering DBSCAN...")
    run_script("dbscan.py", [])

    print("Ejecutando Random Forest sin reducción de dimensionalidad...")
    run_script("rf.py", [])

    print("Comparando métricas...")
    result_dirs = {
        "Base": "results/rf/",
        "PCA": "results/pca/",
        "UMAP": "results/umap/"
    }

    metrics_df = collect_metrics(result_dirs)

    metrics_df.to_csv("results/comparison/metrics_summary.csv", index=False)

    print("Generando gráficos...")
    plot_f1_recall_comparison(metrics_df)

    print("Proceso completado. Gráficos guardados en results/comparison/")
