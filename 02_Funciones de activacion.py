# Ejemplo de una función de activación (Sigmoide)
import numpy as np

# Función sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

entrada = 0.5
activacion = sigmoide(entrada)
print("Valor de activación:", activacion)
