import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ===============================================================
# IMPORT FUNCTIONS AND CONSTANTS FROM THE "Circular_Jesus" MODULE
# ===============================================================
from Circular_Jesus import (
    f_r_func,    # Función de fuerza radial
    delta, R, a, L
)

# ===============================================================
# PHYSICAL PARAMETERS (kept consistent with the "exact" code)
# ===============================================================
M = 2000                    # Total mass in kg
mu0 = 4 * np.pi * 1e-7      # Vacuum permeability (H/m)
rho = M / (np.pi * R**2 * L)  # Mass density (kg/m³)
k_B = 1.380649e-23          # Boltzmann constant (J/K)
n = 1e9                     # Particle density (m⁻³)
P0 = 1e-8                   # Base pressure (Pa)

# Primer término constante
const_term = 1 / np.sqrt(rho * mu0)

# ===============================================================
# DEFINICIÓN DE LA MALLA EN COORDENADAS CILÍNDRICAS (r, φ)
# ===============================================================
Nr = 200    # Número de puntos en r
Nphi = 200  # Número de puntos en φ
r_vals = np.linspace(0, a, Nr)
phi_vals = np.linspace(0, 2 * np.pi, Nphi)
# Creamos la malla: cada fila corresponde a un valor de r y cada columna a un valor de φ
Phi, R_ = np.meshgrid(phi_vals, r_vals)

# ===============================================================
# CÁLCULO DE α(r, φ)
# ===============================================================
alpha_grid = np.zeros_like(R_)

# Para cada punto (r, φ), se integra f_r_func desde 0 hasta r
for i in range(R_.shape[0]):
    for j in range(R_.shape[1]):
        r_val = R_[i, j]
        phi_val = Phi[i, j]
        if r_val == 0:
            alpha_grid[i, j] = const_term
        else:
            integral_fr, err = quad(lambda rp: f_r_func(rp, phi_val), 0, r_val)
            alpha_grid[i, j] = const_term + integral_fr

# ===============================================================
# TRANSFORMACIÓN A COORDENADAS CARTESIANAS PARA VISUALIZACIÓN
# ===============================================================
X = delta * R_ * np.cos(Phi)
Z = R_ * np.sin(Phi)

# ===============================================================
# PLOTEO DE α(r, φ)
# ===============================================================
plt.figure(figsize=(8, 6))
cf = plt.contourf(X, Z, alpha_grid, levels=50, cmap='viridis')
plt.colorbar(cf, label=r"$\alpha(r,\phi)$")
plt.xlabel(r"$x = \delta\, r \cos(\phi)$")
plt.ylabel(r"$z = r \sin(\phi)$")
plt.title(r"Cálculo de $\alpha(r,\phi)$")
plt.axis("equal")
plt.tight_layout()
plt.show()