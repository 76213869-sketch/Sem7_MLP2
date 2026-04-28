# Importamos librerías necesarias
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Dataset XOR
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

# -----------------------------
# 2. Creación del modelo MLP
# -----------------------------
# Red neuronal con:
# - 2 capas ocultas de 4 neuronas cada una
# - Activación no lineal (tanh)
# - Optimizador lbfgs (mejor para pocos datos)
modelo = MLPClassifier(
    hidden_layer_sizes=(4, 4),
    activation='tanh',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

# -----------------------------
# 3. Entrenamiento
# -----------------------------
modelo.fit(X, y)

# -----------------------------
# 4. Predicciones
# -----------------------------
y_pred = modelo.predict(X)

# -----------------------------
# 5. Evaluación
# -----------------------------
print("Predicciones:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred))

print("Matriz de confusión:\n", confusion_matrix(y, y_pred))

print("Reporte:\n", classification_report(y, y_pred))