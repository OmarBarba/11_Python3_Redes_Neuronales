import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar los tamaños de las capas
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicializar los pesos y sesgos
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Calcular la salida de la capa oculta
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Calcular la salida final
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, output, learning_rate):
        # Calcular el error
        error = y - output

        # Calcular el gradiente en la capa final
        delta_output = error * self.sigmoid_derivative(output)

        # Calcular el error en la capa oculta
        error_hidden = delta_output.dot(self.weights_hidden_output.T)

        # Calcular el gradiente en la capa oculta
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # Actualizar los pesos y sesgos
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

# Ejemplo de uso
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Crear una red neuronal con 2 neuronas en la capa oculta
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Entrenar la red neuronal durante 10,000 épocas con una tasa de aprendizaje de 0.1
nn.train(X, y, learning_rate=0.1, num_epochs=10000)

# Realizar predicciones
predictions = nn.forward(X)
print("Predicciones finales:")
print(predictions)
