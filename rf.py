import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix, \
    ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


class BalancedRandomForest:
    def __init__(self, train_file, classes_file, label_column):
        self.train_file = train_file
        self.classes_file = classes_file
        self.label_column = label_column
        self.rf = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_and_prepare_data(self):
        # Leer `VECTOR_train.csv` con delimitador de coma
        train_data = pd.read_csv(self.train_file, delimiter=",")
        # Leer `classes.csv` con delimitador de tubería
        classes_data = pd.read_csv(self.classes_file, delimiter="|")

        # Verificar si las columnas `ID` coinciden
        if "ID" not in train_data.columns or "ID" not in classes_data.columns:
            raise KeyError("La columna 'ID' no existe en uno de los archivos.")

        # Unir ambos conjuntos de datos en la columna `ID`
        data = pd.merge(train_data, classes_data, on="ID")

        # Separar características (X) y etiquetas (y)
        X = data.iloc[:, 1:-3].values  # Excluyendo `ID`, `PCM`, `RS`, y `PU`
        y = data[self.label_column].values
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def balance_data(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train, y_train)

    def train_model(self):
        # Preparar y balancear datos
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()
        X_train_balanced, y_train_balanced = self.balance_data(X_train, y_train)

        # Entrenar modelo
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        self.rf.fit(X_train_balanced, y_train_balanced)

        # Guardar datos de prueba y entrenamiento
        self.X_train, self.X_test, self.y_train, self.y_test = X_train_balanced, X_test, y_train_balanced, y_test

    def evaluate_model(self):
        # Predicciones y métricas
        y_pred = self.rf.predict(self.X_test)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        return y_pred

    def plot_roc_curve(self):
        # Curva ROC
        y_proba = self.rf.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    def plot_confusion_matrix(self, y_pred):
        # Matriz de Confusión
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.rf.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    def plot_feature_importance(self):
        # Importancia de características
        importances = self.rf.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(10), importances[indices[:10]], align="center")
        plt.xticks(range(10), indices[:10])
        plt.xlabel("Feature Index")
        plt.ylabel("Importance Score")
        plt.show()


# Bloque ejecutable
if __name__ == "__main__":
    # Configuración de archivos y columna objetivo
    train_file = "result/VECTOR_train.csv"
    classes_file = "result/Classes.csv"
    label_column = "RS"  # Cambiar a "PCM" u otra etiqueta si es necesario

    # Crear instancia y ejecutar pipeline
    brf = BalancedRandomForest(train_file, classes_file, label_column)
    brf.train_model()
    y_pred = brf.evaluate_model()

    # Generar gráficos
    print("\nGenerando gráficas...")
    brf.plot_roc_curve()
    brf.plot_confusion_matrix(y_pred)
    brf.plot_feature_importance()
