import preprocess
import vectorize
import subprocess
import os
import pandas as pd

def run_script(script_name, args):
    cmd = ["python", script_name] + args
    subprocess.run(cmd, check=True)

def collect_metrics(result_dirs):
    metrics = []
    for method, path in result_dirs.items():
        metrics_path = os.path.join(path, "metrics.txt")
        with open(metrics_path, "r") as f:
            lines = f.readlines()
            accuracy = float(lines[0].split(":")[1].strip())
            metrics.append({"method": method, "accuracy": accuracy})
    return pd.DataFrame(metrics)


# Crear carpetas necesarias (data, result)
if not os.path.exists("data/"):
    os.makedirs("data/")
if not os.path.exists("result/"):
    os.makedirs("result/")
if not os.path.exists("model/"):
    os.makedirs("model/")

# PREPROCESADO
input_file = 'data/DataI_MD.csv'  # <--- Añadir aquí la ruta al archivo CSV original!!
output_file = 'data/DataI_MD_POST.csv'
preprocess.preprocess(input_file, output_file)

# VECTORIZACIÓN
vector_input = output_file
vector_output = 'result/DataI_MD_VECTOR.csv'
# vectorize(vector_input, vector_output)  # Descomenta si quieres ejecutar esta parte.

# Limpiar los datos manteniendo las etiquetas
vector_data = pd.read_csv(vector_output, delimiter="|")

# Guardar las clases en un archivo separado
df_classes = vector_data[["ID", "PCM", "RS", "PU"]]
df_classes.to_csv('result/Classes.csv', sep="|", index=False)

# Mantener todas las etiquetas en los datos limpios
vector_data_cleaned = vector_data.drop(columns=["TXT"], errors='ignore')
cleaned_output = 'result/DataI_MD_VECTOR_CLEAN.csv'
vector_data_cleaned.to_csv(cleaned_output, sep="|", index=False)

# Dividir en conjunto de entrenamiento y prueba
train_file = 'result/VECTOR_train.csv'
test_file = 'result/VECTOR_test.csv'
preprocess.divide_csv(cleaned_output, train_file, test_file, 0.85, delimiter="|")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)

    print("Ejecutando reducción dimensional t-SNE...")
    run_script("rf-tsne.py", [])

    print("Ejecutando reducción dimensional UMAP...")
    run_script("rf-umap.py", [])

    print("Ejecutando clustering DBSCAN...")
    run_script("dbscan.py", [])

    print("Ejecutando Random Forest con UMAP...")
    run_script("rf.py", [])

    print("Comparando métricas...")
    result_dirs = {
        "t-SNE": "results/tsne/",
        "UMAP": "results/umap/",
        "DBSCAN": "results/dbscan/"
    }
    metrics_df = collect_metrics(result_dirs)
    metrics_df.to_csv("results/comparison/metrics_summary.csv", index=False)
    print("Métricas guardadas en results/comparison/metrics_summary.csv")

# TOKENIZADO + VECTORIZACIÓN
#input_csv_path = 'data/DataI_MD_POST.csv'
#output_csv_path = 'data/DataI_MD_VECTOR.csv'
# Se puede especificar el modelo (por defecto: AIDA-UPM/BERTuit-base),
# from_tf (por defecto: True), la columna de procesado del csv de entrada (por defecto: 4)
# y el batch_size para repartir la carga de trabajo (por defecto: 64)
#vectorize.vectorize(input_csv_path, output_csv_path, True)
#preprocess.divide_csv('data/DataI_MD_VECTOR.csv', 'data/VECTOR_BERTuit90%.csv', 'data/VECTOR_test.csv', 0.9)
#preprocess.divide_csv('data/DataI_MD_POST.csv', 'data/DataI_MD_POST90%.csv', 'data/DataI_MD_POST10%.csv', 0.9, delimiter="|")