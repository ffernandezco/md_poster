import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
from tqdm import tqdm
import os


def load_model_and_tokenizer(model_name):
    """
    Carga el modelo y tokenizer de Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def generate_embeddings(texts, tokenizer, model, device):
    """
    Genera embeddings a partir de textos utilizando un modelo de Hugging Face.
    """
    embeddings = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for text in tqdm(texts, desc="Generando embeddings"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            # Usamos el embedding del token [CLS]
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())
    return embeddings


def save_embeddings_to_csv(dataframe, embeddings, output_file):
    """
    Combina los embeddings con el dataframe original y guarda en un CSV.
    """
    embedding_df = pd.DataFrame(embeddings)
    result = pd.concat([dataframe, embedding_df], axis=1)
    result.to_csv(output_file, index=False, sep="|")
    print(f"Embeddings guardados en {output_file}")


def main(input_file, model_name, output_file, device):
    """
    Flujo principal del script.
    """
    # Carga de datos
    print(f"Cargando datos desde {input_file}...")
    df = pd.read_csv(input_file, sep="|", header=None, names=["ID", "PCM", "RS", "PU", "TXT"])

    # Carga del modelo y tokenizer
    print(f"Cargando modelo {model_name}...")
    tokenizer, model = load_model_and_tokenizer(model_name)

    # Generación de embeddings
    print("Generando embeddings...")
    texts = df["TXT"].fillna("").tolist()
    embeddings = generate_embeddings(texts, tokenizer, model, device)

    # Guardar embeddings
    save_embeddings_to_csv(df, embeddings, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para generar embeddings usando modelos de Hugging Face.")
    parser.add_argument("--input_file", type=str, required=True, help="Ruta al archivo CSV preprocesado.")
    parser.add_argument("--model_name", type=str, default="vinai/bertweet-base", help="Nombre del modelo Hugging Face a utilizar.")
    parser.add_argument("--output_file", type=str, required=True, help="Ruta al archivo CSV de salida.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo para ejecutar el modelo (cpu o cuda).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    main(args.input_file, args.model_name, args.output_file, args.device)
