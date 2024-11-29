# Clustering de textos mediante DBSCAN
**Francisco Fernández Condado**

Minería de Datos / Grado en Ingeniería Informática de Gestión y Sistemas de Información

_[UPV-EHU](https://www.ehu.eus) / Curso 2024-25_

## Introducción
Partiendo de un conjunto de datos como _DataI_, este proyecto permite realizar un preprocesamiento de los datos, así como la tokenización y vectorización usando cualquier modelo de Hugging Face. Usando una implementación de Random Forest con SMOTE para reducir el desbalanceo, permite obtener gráficos y métricas según la técnica de reducción de dimensionalidad utilizada, permitiendo comparar si PCA o UMAP ofrecen mejores resultados.

## Bibliotecas necesarios para la ejecución completa
- [os](https://docs.python.org/3/library/os.html)
- [argparse](https://docs.python.org/3/library/argparse.html)
- [csv](https://docs.python.org/3/library/csv.html)
- [statistics](https://docs.python.org/3/library/statistics.html)
- [subprocess](https://docs.python.org/3/library/subprocess.html)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [imbalanced-learn (imblearn)](https://imbalanced-learn.org/stable/)
- [UMAP-learn](https://umap-learn.readthedocs.io/en/latest/)
- [transformers](https://huggingface.co/docs/transformers/)
- [torch](https://pytorch.org/)
- [tqdm](https://tqdm.github.io/)
- [numpy](https://numpy.org/)

## Instrucciones de uso
El fichero `main.py` incluye la generación automatizada de resultados. Para poderlo generar, se debe contar con un fichero que incluya los embeddings en la ruta `result/DataI_MD_VECTOR.csv`. Tras la ejecución, se creará el directorio `results`, que incluye los resultados tanto utilizando como sin utilizar las técnicas de reducción de dimensionalidad, así como la matriz de confusión asociada. El directorio `results/comparison` permite además comparar todas las métricas obtenidas de forma conjunta, ofreciendo gráficos descriptivos.

## Generación de vectorizaciones
En el directorio `vectores` se han incluido algunos vectores ya generados a partir del conjunto de datos. Se puede copiar como `result/DataI_MD_VECTOR.csv` para ejecutar el trabajo de `main.py`. No obstante, si se desea generar un nuevo vector a partir de un modelo personalizado de Hugging Face, puede hacerse copiando en primer lugar en `data/DataI_MD.csv` el fichero original, ejecutando `preprocess.py` y el siguiente comando:

```bash
python vectorize.py \
    --input_file data/DataI_MD_POST.csv \
    --model_name ... \      # Sustituir '...' por el modelo, por ejemplo vinai/bertweet-base
    --output_file result/DataI_MD_VECTOR.csv \
    --device cpu            # Cambiar por CUDA si se tiene gráfica dedicada