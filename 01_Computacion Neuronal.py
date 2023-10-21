
# Ejemplo de una neurona artificial
import numpy as np

# Función que simula la computación de una neurona
def neurona(entradas, pesos):
    # Realizar el cálculo de la suma ponderada
    suma_ponderada = np.dot(entradas, pesos)
    return suma_ponderada

# Entradas de ejemplo y pesos
entradas = np.array([2, 3, 1])
pesos = np.array([0.5, -0.2, 0.1])

resultado = neurona(entradas, pesos)
print("Resultado de la neurona:", resultado)
