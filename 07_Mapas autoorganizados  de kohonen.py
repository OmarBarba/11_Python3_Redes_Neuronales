import numpy as np

class KohonenMap:
    def __init__(self, input_size, map_size):
        self.input_size = input_size
        self.map_size = map_size
        self.weights = np.random.rand(map_size, map_size, input_size)

    def find_best_matching_unit(self, input_vector):
        # Calcula la distancia entre el vector de entrada y todos los pesos
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        # Encuentra la unidad de mapeo con la distancia más pequeña
        bmu = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu

    def update_weights(self, input_vector, bmu, learning_rate, radius):
        for i in range(self.map_size):
            for j in range(self.map_size):
                distance = np.linalg.norm(np.array([i, j]) - np.array(bmu))
                if distance <= radius:
                    # Actualiza los pesos de las unidades cercanas
                    influence = np.exp(-distance / (2 * radius**2))
                    self.weights[i, j, :] += learning_rate * influence * (input_vector - self.weights[i, j, :])

    def train(self, data, num_epochs, initial_learning_rate, initial_radius):
        for epoch in range(num_epochs):
            learning_rate = initial_learning_rate * (1 - epoch / num_epochs)
            radius = initial_radius * (1 - epoch / num_epochs)
            for input_vector in data:
                bmu = self.find_best_matching_unit(input_vector)
                self.update_weights(input_vector, bmu, learning_rate, radius)

# Ejemplo de uso
# Genera datos de ejemplo
data = np.random.rand(100, 2)
# Normaliza los datos en el rango [0, 1]
data = (data - data.min()) / (data.max() - data.min())

# Crea un mapa SOM de 5x5 con vectores de entrada de tamaño 2
map_size = 5
input_size = 2
smap = KohonenMap(input_size, map_size)

# Entrena el mapa SOM
smap.train(data, num_epochs=100, initial_learning_rate=0.1, initial_radius=map_size / 2)

# Visualiza los resultados
import matplotlib.pyplot as plt
plt.scatter(data[:, 0], data[:, 1])
for i in range(map_size):
    for j in range(map_size):
        plt.scatter(smap.weights[i, j, 0], smap.weights[i, j, 1], color='red', marker='x')
plt.title("Mapa Autoorganizado de Kohonen")
plt.show()
