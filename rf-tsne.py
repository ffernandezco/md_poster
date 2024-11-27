import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class BalancedRandomForest:
    def __init__(self, train_file, label_column="RS"):
        self.train_file = train_file
        self.label_column = label_column
        self.rf = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

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

        # Aplicar PCA para reducir dimensionalidad (opcional)
        print("Aplicando PCA para reducir dimensionalidad...")
        pca = PCA(n_components=10, random_state=42)
        X_train_resampled = pca.fit_transform(X_train_resampled)
        X_test = pca.transform(X_test)

        return X_train_resampled, X_test, y_train_resampled, y_test

    def train_model(self):
        print("Dividiendo los datos...")
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_prepare_data()

        print("Entrenando el modelo Random Forest...")
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            class_weight="balanced"
        )
        self.rf.fit(self.X_train, self.y_train)
        print("Entrenamiento completado.")

    def evaluate_model(self, output_dir="results/pca/"):
        os.makedirs(output_dir, exist_ok=True)
        print("Evaluando el modelo...")
        y_pred = self.rf.predict(self.X_test)

        # Guardar resultados
        report = classification_report(self.y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(self.y_test, y_pred)

        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Classification Report:\n{classification_report(self.y_test, y_pred)}\n")

        print("Reporte de clasificación guardado.")
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Matriz de Confusión")
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

# Ejecución
if __name__ == "__main__":
    brf = BalancedRandomForest(
        train_file="result/VECTOR_train.csv",
        label_column="RS"
    )
    brf.train_model()
    brf.evaluate_model()
