# Ejemplo de datos linealmente separables
import matplotlib.pyplot as plt

# Datos de ejemplo
x = [1, 2, 3, 4, 5]
y = [0, 0, 1, 1, 1]

plt.scatter(x, y)
plt.xlabel("Característica X")
plt.ylabel("Clase (0 o 1)")
plt.title("Datos Linealmente Separables")
plt.show()
