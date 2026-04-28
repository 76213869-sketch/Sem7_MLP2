# Importamos las librerías necesarias
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Definición del dataset XOR
# -----------------------------
# Entradas (features)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Salidas esperadas (labels)
y = np.array([0, 1, 1, 0])

# -----------------------------
# 2. Creación del modelo
# -----------------------------
# Se crea un perceptrón simple (modelo lineal)
modelo = Perceptron(max_iter=1000, tol=1e-3)

# -----------------------------
# 3. Entrenamiento del modelo
# -----------------------------
# El modelo aprende a partir de los datos
modelo.fit(X, y)

# -----------------------------
# 4. Predicciones
# -----------------------------
# El modelo intenta predecir los mismos datos
y_pred = modelo.predict(X)

# -----------------------------
# 5. Evaluación del modelo
# -----------------------------
# Exactitud (accuracy)
print("Predicciones:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred))

# Matriz de confusión
print("Matriz de confusión:\n", confusion_matrix(y, y_pred))

# Reporte completo (precision, recall, f1-score)
print("Reporte:\n", classification_report(y, y_pred))