##########################################################
###############HAMING###################################

print("Haming")

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
print("Hopefield")

import numpy as np

class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            weight = np.outer(pattern, pattern)
            np.fill_diagonal(weight, 0)
            self.weights += weight

    def recall(self, input_pattern, max_iterations=100):
        current_pattern = input_pattern
        for _ in range(max_iterations):
            new_pattern = np.sign(np.dot(self.weights, current_pattern))
            if np.array_equal(new_pattern, current_pattern):
                return new_pattern
            current_pattern = new_pattern
        return None  # La red no ha convergido

# Ejemplo de uso
pattern_size = 4

# Crear una instancia de la Red de Hopfield
hopfield_net = HopfieldNetwork(pattern_size)

# Patrones de ejemplo (en forma de lista de listas)
patterns = [
    [1, -1, 1, -1],
    [-1, -1, 1, 1],
    [1, 1, -1, -1]
]

# Entrenar la red con los patrones
hopfield_net.train(patterns)

# Patrón de consulta
query_pattern = [1, 1, 1, -1]

# Realizar una consulta
retrieved_pattern = hopfield_net.recall(query_pattern)

# Imprimir el patrón recuperado
if retrieved_pattern is not None:
    print("Patrón recuperado:", retrieved_pattern)
else:
    print("La red no ha convergido después de un número máximo de iteraciones.")

#######################################################
#####################Hebb#########################
print("Hebb")



class HebbianNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((input_size, output_size))

    def train(self, input_data, output_data):
        for i in range(self.input_size):
            for j in range(self.output_size):
                self.weights[i][j] += input_data[i] * output_data[j]

    def predict(self, input_data):
        output_data = np.dot(input_data, self.weights)
        return output_data

# Ejemplo de uso
input_size = 3
output_size = 2

# Crear una instancia de la Red de Hebb
hebb_net = HebbianNetwork(input_size, output_size)

# Datos de ejemplo para entrenar
input_data = np.array([[1, 0, 1],
                      [0, 1, 1],
                      [1, 1, 0]])

output_data = np.array([[1, 0],
                       [0, 1],
                       [1, 1]])

# Entrenar la red utilizando la regla de Hebb
for i in range(len(input_data)):
    hebb_net.train(input_data[i], output_data[i])

# Patrón de entrada para predecir
input_pattern = np.array([1, 0, 1])

# Realizar una predicción
predicted_output = hebb_net.predict(input_pattern)

print("Patrón de entrada:", input_pattern)
print("Patrón de salida predicho:", predicted_output)



#######################################################
#####################Boltzmann#########################
print("Boltzmann")

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.random.randn(num_visible, num_hidden)
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_hidden(self, visible_data):
        hidden_prob = self.sigmoid(np.dot(visible_data, self.weights) + self.hidden_bias)
        hidden_states = hidden_prob > np.random.rand(self.num_hidden)
        return hidden_prob, hidden_states

    def sample_visible(self, hidden_data):
        visible_prob = self.sigmoid(np.dot(hidden_data, self.weights.T) + self.visible_bias)
        visible_states = visible_prob > np.random.rand(self.num_visible)
        return visible_prob, visible_states

    def train(self, data, learning_rate=0.1, num_epochs=100, batch_size=10):
        for epoch in range(num_epochs):
            np.random.shuffle(data)
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                visible_data = batch
                hidden_prob, hidden_states = self.sample_hidden(visible_data)

                # Gibb's sampling
                for _ in range(10):
                    visible_prob, visible_states = self.sample_visible(hidden_states)
                    hidden_prob, hidden_states = self.sample_hidden(visible_states)

                # Update weights and biases
                positive_association = np.dot(visible_data.T, hidden_prob)
                negative_association = np.dot(visible_states.T, hidden_states)

                self.weights += learning_rate * (positive_association - negative_association) / batch_size
                self.visible_bias += learning_rate * np.mean(visible_data - visible_states, axis=0)
                self.hidden_bias += learning_rate * np.mean(hidden_prob - hidden_states, axis=0)

    def generate_samples(self, num_samples):
        samples = np.zeros((num_samples, self.num_visible))
        for _ in range(10):
            _, hidden_states = self.sample_hidden(samples)
            _, samples = self.sample_visible(hidden_states)
        return samples

# Ejemplo de uso
num_visible = 6
num_hidden = 2

# Crear una instancia de RBM
rbm = RBM(num_visible, num_hidden)

# Datos de entrenamiento de ejemplo (binarios)
data = np.array([[1, 0, 1, 0, 1, 0],
                 [1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 0],
                 [0, 1, 0, 0, 0, 1]])

# Entrenar la RBM
rbm.train(data, num_epochs=1000, batch_size=2)

# Generar muestras con la RBM entrenada
generated_samples = rbm.generate_samples(num_samples=5)

print("Muestras generadas:")
print(generated_samples)
