import preprocess
import vectorize
import os
import pandas as pd


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
preprocess.divide_csv(cleaned_output, train_file, test_file, 0.9, delimiter="|")


# TOKENIZADO + VECTORIZACIÓN
#input_csv_path = 'data/DataI_MD_POST.csv'
#output_csv_path = 'data/DataI_MD_VECTOR.csv'
# Se puede especificar el modelo (por defecto: AIDA-UPM/BERTuit-base),
# from_tf (por defecto: True), la columna de procesado del csv de entrada (por defecto: 4)
# y el batch_size para repartir la carga de trabajo (por defecto: 64)
#vectorize.vectorize(input_csv_path, output_csv_path, True)
#preprocess.divide_csv('data/DataI_MD_VECTOR.csv', 'data/VECTOR_BERTuit90%.csv', 'data/VECTOR_test.csv', 0.9)
#preprocess.divide_csv('data/DataI_MD_POST.csv', 'data/DataI_MD_POST90%.csv', 'data/DataI_MD_POST10%.csv', 0.9, delimiter="|")