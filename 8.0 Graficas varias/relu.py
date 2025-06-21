import numpy as np
import matplotlib.pyplot as plt

# Dominio
x = np.linspace(-10, 10, 400)
y = np.maximum(0, x)          # ReLU(z) = max(0, z)

# Gráfico
plt.figure()
plt.plot(x, y)                # un solo trazo, sin especificar color
plt.title("Función ReLU")
plt.xlabel("z")
plt.ylabel("ReLU(z)")
plt.grid(True)

# Guardar y mostrar
plt.savefig("8.0 Graficas varias/relu.png", dpi=300, bbox_inches="tight")
plt.show()
