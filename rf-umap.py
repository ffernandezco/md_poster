from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from umap import UMAP
import pandas as pd


def evaluate_class_1_f1(y_test, y_prob, threshold_range=np.linspace(0.1, 0.99, 50)):
    """
    Encuentra el umbral que maximiza el F1-score para la clase minoritaria.
    """
    best_f1 = 0
    best_threshold = 0
    for threshold in threshold_range:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

class BalancedRandomForestUMAP:
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

        # Aplicar UMAP para reducir dimensionalidad
        print("Aplicando UMAP para reducir dimensionalidad")
        umap = UMAP(n_components=5, random_state=42)  # Reduce a 5 dimensiones
        X_train_resampled = umap.fit_transform(X_train_resampled)
        X_test = umap.transform(X_test)

        return X_train_resampled, X_test, y_train_resampled, y_test

    def train_model(self):
        print("Dividiendo los datos...")
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_prepare_data()

        print("Entrenando el modelo Random Forest...")
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight={0: 1, 1: 10}  # Penaliza más los errores en la clase 1
        )
        self.rf.fit(self.X_train, self.y_train)
        print("Entrenamiento completado.")

    def evaluate_model(self):
        print("Evaluando el modelo...")
        y_prob = self.rf.predict_proba(self.X_test)[:, 1]

        # Optimizar el F1-score de la clase minoritaria
        best_threshold, best_f1 = evaluate_class_1_f1(self.y_test, y_prob)
        print(f"Umbral óptimo para F1-score clase 1: {best_threshold}, F1: {best_f1}")

        y_pred = (y_prob >= best_threshold).astype(int)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Matriz de Confusión")
        plt.show()


# Ejecución
if __name__ == "__main__":
    brf = BalancedRandomForestUMAP(
        train_file="result/VECTOR_train.csv",
        label_column="RST"  # Etiqueta a analizar
    )
    brf.train_model()
    brf.evaluate_model()
