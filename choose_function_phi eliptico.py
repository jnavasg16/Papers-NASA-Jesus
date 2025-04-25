import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Valores de sigma
sigma_values = [1, 2, 3, 4, 5]

# Valores de phi
phi_values = np.linspace(-2 * np.pi, 4 * np.pi, 1000)  # Extendiendo para ver más periodos

# Crear el plot
plt.figure(figsize=(10, 6))
for sigma in sigma_values:
    y_vals = np.exp(-sigma * (phi_values - np.pi)**2)
    plt.plot(phi_values, y_vals, label=f"$\\sigma = {sigma}$")

# Personalizar el gráfico
plt.title("Graph of $e^{-\\sigma (\\varphi - \\pi)^2}$ for different values of \\sigma.")
plt.xlabel("$\\varphi$")
plt.ylabel("$e^{-\\sigma (\\varphi - \\pi)^2}$")
plt.legend()
plt.grid()
plt.show()

# Función para calcular la serie de Fourier
def fourier_series(func, T, n_terms, x_vals):
    a0 = (2 / T) * quad(lambda x: func(x), 0, T)[0]
    fourier_approx = a0 / 2
    print(f"a0/2: {a0 / 2}")

    for n in range(1, n_terms + 1):
        an = (2 / T) * quad(lambda x: func(x) * np.cos(2 * np.pi * n * x / T), 0, T)[0]
        bn = (2 / T) * quad(lambda x: func(x) * np.sin(2 * np.pi * n * x / T), 0, T)[0]
        fourier_approx += an * np.cos(2 * np.pi * n * x_vals / T) + bn * np.sin(2 * np.pi * n * x_vals / T)
        print(f"n={n}: an={an}, bn={bn}")
    
    return fourier_approx

# Parámetros
sigma = 5
T = 2 * np.pi  # Periodo de la función
n_terms = 7  # Grado de la serie de Fourier
x_vals = np.linspace(- T, 2 * T, 1000)  # Múltiples períodos para visualizar mejor
func = lambda phi: np.exp(-sigma * (phi - np.pi)**2)  # Nueva función con el cambio de variable

# Calcular la serie de Fourier
fourier_approximation = fourier_series(func, T, n_terms, x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, func(x_vals), label="Original function: $e^{-\\sigma (\\varphi - \\pi)^2}$", color="blue", linewidth=2)
plt.plot(x_vals, fourier_approximation, label=f"Fourier Series ({n_terms} terms)", color="orange", linestyle="--", linewidth=2)
plt.title(f"Comparison of $e^{{-\\sigma (\\varphi - \\pi)^2}}$ and its Fourier Series ($\\sigma={sigma}$)")
plt.xlabel("$\\varphi$")
plt.ylabel("$f(\\varphi)$")
plt.legend()
plt.grid()
plt.show()
