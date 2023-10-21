# Ejemplo de un perceptrón Adeline

print("Perceptron adeline")

import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.cost_history = []

        for _ in range(self.n_iterations):
            output = self.net_input(X)
            errors = (y - output)
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_history.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

# Ejemplo de uso
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, 1, -1, -1])

adaline = Adaline(learning_rate=0.01, n_iterations=100)
adaline.fit(X, y)

print("Pesos finales:", adaline.weights[1:])
print("Término de sesgo (bias) final:", adaline.weights[0])

############perceptron madeline################

print("Perceptron Madeline")

class Madaline:
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1])
        self.cost_history = []

        for _ in range(self.n_iterations):
            cost = 0
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = target - output
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error
                cost += 0.5 * error**2
            self.cost_history.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)

# Ejemplo de uso
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([1, 1, -1, -1])

madaline = Madaline(learning_rate=0.01, n_iterations=100)
madaline.fit(X, y)

print("Pesos finales:", madaline.weights[1:])
print("Término de sesgo (bias) final:", madaline.weights[0])
