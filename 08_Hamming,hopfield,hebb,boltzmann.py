##########################################################
###############HAMING###################################
import numpy as np

class HammingNetwork:
    def __init__(self, pattern_size, threshold):
        self.pattern_size = pattern_size
        self.threshold = threshold
        self.weights = np.zeros((pattern_size, pattern_size))

    def train(self, patterns):
        for pattern in patterns:
            weight = np.outer(pattern, pattern)
            np.fill_diagonal(weight, 0)
            self.weights += weight

    def recall(self, input_pattern):
        result = np.sign(np.dot(input_pattern, self.weights) - self.threshold)
        return result

# Ejemplo de uso
pattern_size = 8
threshold = 0.0

# Crear una instancia de la Red de Hamming
hamming_net = HammingNetwork(pattern_size, threshold)

# Patrones de ejemplo (en forma de lista de listas)
patterns = [
    [1, -1, -1, 1, 1, -1, 1, -1],
    [-1, -1, 1, -1, 1, -1, 1, 1],
    [1, 1, -1, -1, 1, -1, -1, -1]
]

# Entrenar la red con los patrones
hamming_net.train(patterns)

# Patrón de consulta
query_pattern = [1, -1, -1, 1, 1, -1, 1, -1]

# Realizar una consulta
retrieved_pattern = hamming_net.recall(query_pattern)

# Imprimir el patrón recuperado
print("Patrón recuperado:", retrieved_pattern)

#######################################################
#####################HOPEFIELD#########################

