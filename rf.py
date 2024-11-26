import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import os


class BalancedRandomForest:
    def __init__(self, train_file, label_column="RS", results_dir="results"):
        self.train_file = train_file
        self.label_column = label_column
        self.results_dir = results_dir
        self.rf = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        # Crear directorio de resultados si no existe
        os.makedirs(self.results_dir, exist_ok=True)

    def load_and_prepare_data(self):
        print("Cargando y preparando los datos...")
        train_data = pd.read_csv(self.train_file, delimiter=",")

        # Validar existencia de las columnas requeridas
        if self.label_column not in train_data.columns:
            raise KeyError(f"La etiqueta '{self.label_column}' no existe en los datos.")

        # Separar características (X) y etiquetas (y)
        X = train_data.drop(columns=["ID", "PCM", "RS", "PU"], errors='ignore').values
        y = train_data[self.label_column].values

        # Dividir los datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
        print("Aplicando SMOTE para balancear las clases...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        return X_train_resampled, X_test, y_train_resampled, y_test

    def train_model(self):
        print("Dividiendo los datos...")
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_prepare_data()

        print("Entrenando el modelo Random Forest...")
        self.rf = RandomForestClassifier(
            n_estimators=50,  # Reducir para mayor velocidad
            max_depth=10,  # Limitar profundidad de árboles
            random_state=42,
            class_weight="balanced"  # Ajustar pesos automáticamente
        )
        self.rf.fit(self.X_train, self.y_train)
        print("Entrenamiento completado.")

    def evaluate_model(self):
        print("Evaluando el modelo...")
        y_prob = self.rf.predict_proba(self.X_test)[:, 1]  # Probabilidad para la clase 1
        threshold = 0.6  # Ajustar el umbral
        y_pred = (y_prob >= threshold).astype(int)

        # Calcular métricas
        accuracy = accuracy_score(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        # Guardar métricas en archivos
        print("Guardando resultados...")
        metrics_file = os.path.join(self.results_dir, "metrics.txt")
        cm_file = os.path.join(self.results_dir, "confusion_matrix.png")
        report_file = os.path.join(self.results_dir, "classification_report.csv")

        # Guardar accuracy y clasificación en texto
        with open(metrics_file, "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write("Classification Report:\n")
            f.write(classification_report(self.y_test, y_pred))

        # Guardar matriz de confusión como imagen
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Matriz de Confusión")
        plt.savefig(cm_file)
        plt.close()

        # Guardar classification report en CSV
        report_df = pd.DataFrame(class_report).transpose()
        report_df.to_csv(report_file, index=True)
        print("Resultados guardados en:", self.results_dir)


# Ejecución
if __name__ == "__main__":
    brf = BalancedRandomForest(
        train_file="result/VECTOR_train.csv",
        label_column="RS",  # Fijar etiqueta a analizar
        results_dir="results/rf"
    )
    brf.train_model()
    brf.evaluate_model()
