import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE


class BalancedRandomForestTSNE:
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

        # Aplicar t-SNE después de SMOTE
        print("Aplicando t-SNE para reducir dimensionalidad")
        tsne = TSNE(n_components=2, random_state=42)  # Reducir a 2 dimensiones
        X_train_resampled = tsne.fit_transform(X_train_resampled)
        X_test = tsne.fit_transform(X_test)  # Aplicar también al conjunto de prueba (subóptimo)

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
        # Obtener probabilidades y calcular la curva precision-recall
        y_prob = self.rf.predict_proba(self.X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_prob)

        # Determinar el umbral óptimo basado en F1-score
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_threshold = thresholds[f1_scores.argmax()]
        print(f"Umbral óptimo basado en F1-score: {optimal_threshold}")

        # Aplicar el umbral óptimo
        y_pred = (y_prob >= optimal_threshold).astype(int)

        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

        # Mostrar matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Matriz de Confusión")
        plt.show()

# Ejecución
if __name__ == "__main__":
    brf = BalancedRandomForestTSNE(
        train_file="result/VECTOR_train.csv",
        label_column="RS"  # Fijar etiqueta a analizar
    )
    brf.train_model()
    brf.evaluate_model()
