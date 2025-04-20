import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy as sp

# ===============================================================
# IMPORT FUNCTIONS AND CONSTANTS FROM THE "Circular_Jesus" MODULE
# ===============================================================
from Circular_Jesus import (
    jr_func, jphi_func, jy_func,
    Bphi_func, By_func,
    f_r_func, f_y_func, f_phi_func,
    delta, R, a, L
)

# ===============================================================
# PHYSICAL PARAMETERS (kept consistent with the "exact" code)
# ===============================================================
M = 2000                    # Total mass in kg
mu0 = 4 * np.pi * 1e-7      # Vacuum permeability (H/m)
rho = M / (np.pi * R**2 * L) # Mass density (kg/m³)
alpha = np.sqrt(1 / (rho * mu0))  # Factor α (m/s)
k_B = 1.380649e-23          # Boltzmann constant (J/K)
n = 1e9                     # Particle density (m⁻³) for temperature calculations
P0 = 1e-8                   # Base pressure (Pa)

# ===============================================================
# DEFINITION OF THE MESH IN CYLINDRICAL COORDINATES
# ===============================================================
phi_vals = np.linspace(0, 2*np.pi, 200)
r_vals = np.linspace(1e-1, a, 200)
Phi, R_ = np.meshgrid(phi_vals, r_vals)

# Transformation to "cartesian" coordinates (X, Z) for visualization
X = delta * R_ * np.cos(Phi)
Z = R_ * np.sin(Phi)

# ===============================================================
# CALCULATION OF THE MAGNETIC FIELD
# ===============================================================
# It is assumed that the field has components B^φ and B^y (B^r = 0)
Bphi_vals = Bphi_func(R_, Phi)  # Azimuthal component B^φ
By_vals   = By_func(R_, Phi)    # Axial component B^y

# ===============================================================
# CALCULATION OF THE PRESSURE DISTRIBUTION
# P(r,φ) = P0 + ∫₀^r f^r(r',φ) dr' + ∫₀^r [r'*B^φ(r',φ)^2/μ0] dr'
# ===============================================================
def pressure(r_val, phi_val):
    # First integral: ∫₀^r f^r(r', φ) dr'
    integral_fr, err_fr = quad(lambda rp: f_r_func(rp, phi_val), 0, r_val)
    # Second integral: ∫₀^r [rp*(B^φ(rp, φ)^2)/μ0] dr'
    integral_Bphi, err_Bphi = quad(lambda rp: rp * (Bphi_func(rp, phi_val)**2) / mu0, 0, r_val)
    return P0 + integral_fr + integral_Bphi

# Create the pressure mesh evaluated at each point (r, φ)
pressure_vals = np.zeros_like(R_)
for i in range(R_.shape[0]):
    for j in range(R_.shape[1]):
        pressure_vals[i, j] = pressure(R_[i, j], Phi[i, j])

# ===============================================================
# CALCULATION OF THE TEMPERATURE DISTRIBUTION
# T(r,φ) = [P(r,φ) - |B(r,φ)|^2/μ0] / (n*k_B)
# ===============================================================
# |B|^2 = (B^φ)^2 + (B^y)^2
B_mod_sq = Bphi_vals**2 + By_vals**2
temperature_vals = (pressure_vals - B_mod_sq / mu0) / (n * k_B)

# ===============================================================
# JOINT PLOT OF PRESSURE AND TEMPERATURE
# ===============================================================
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Pressure subplot
cp0 = axs[0].contourf(X, Z, pressure_vals, cmap='inferno', levels=50)
axs[0].set_title("Pressure Distribution")
axs[0].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[0].set_ylabel("$z = r \\sin(\\phi)$")
axs[0].axis('equal')
fig.colorbar(cp0, ax=axs[0], label="Pressure (Pa)")

# Temperature subplot
cp1 = axs[1].contourf(X, Z, temperature_vals, cmap='plasma', levels=50)
axs[1].set_title("Temperature Distribution")
axs[1].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[1].set_ylabel("$z = r \\sin(\\phi)$")
axs[1].axis('equal')
fig.colorbar(cp1, ax=axs[1], label="Temperature (K)")

plt.tight_layout()
plt.show()

# ===============================================================
# CALCULATION OF THE VELOCITY FIELD: v = α * B
# ===============================================================
# It is assumed that B_r = 0, so v_r = 0.
v_r   = np.zeros_like(Bphi_vals)    # Radial component v_r
v_phi = alpha * Bphi_vals             # Azimuthal component v_φ
v_y   = alpha * By_vals               # Axial component v_y

# ===============================================================
# JOINT PLOT OF THE VELOCITY COMPONENTS
# ===============================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

cp_vr = axs[0].contourf(X, Z, v_r, cmap='viridis', levels=50)
axs[0].set_title("$v_r$ (Radial Component)")
axs[0].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[0].set_ylabel("$z = r \\sin(\\phi)$")
axs[0].axis("equal")
fig.colorbar(cp_vr, ax=axs[0], label="$v_r$")

cp_vphi = axs[1].contourf(X, Z, v_phi, cmap='viridis', levels=50)
axs[1].set_title("$v_\\phi$ (Azimuthal Component)")
axs[1].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[1].set_ylabel("$z = r \\sin(\\phi)$")
axs[1].axis("equal")
fig.colorbar(cp_vphi, ax=axs[1], label="$v_\\phi$")

cp_vy = axs[2].contourf(X, Z, v_y, cmap='viridis', levels=50)
axs[2].set_title("$v_y$ (Axial Component)")
axs[2].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[2].set_ylabel("$z = r \\sin(\\phi)$")
axs[2].axis("equal")
fig.colorbar(cp_vy, ax=axs[2], label="$v_y$")

plt.tight_layout()
plt.show()

# ===============================================================
# ANÁLISIS SINTÉTICO: TRAYECTORIA DEL SATÉLITE
# ===============================================================
from scipy.interpolate import RegularGridInterpolator

# Definir la trayectoria del satélite en el plano (x, z)
x_start, x_end = -delta * a, delta * a  # Puntos de entrada y salida
z_fixed = 0.0                           # Fijamos z = 0 (puedes cambiarlo)
n_points = 100                          # Número de puntos en la trayectoria
x_traj = np.linspace(x_start, x_end, n_points)  # Coordenadas x
z_traj = np.zeros_like(x_traj)                  # Coordenadas z (constante)

# Convertir la trayectoria (x, z) a coordenadas cilíndricas (r, φ)
r_traj = np.sqrt((x_traj / delta)**2 + z_traj**2)  # r = sqrt((x/δ)² + z²)
phi_traj = np.arctan2(z_traj, x_traj / delta)      # φ = arctan(z / (x/δ))
phi_traj = np.mod(phi_traj + 2 * np.pi, 2 * np.pi)

# Crear interpoladores para los campos calculados
interp_v_r = RegularGridInterpolator((r_vals, phi_vals), v_r, bounds_error=False, fill_value=0)
interp_v_phi = RegularGridInterpolator((r_vals, phi_vals), v_phi, bounds_error=False, fill_value=0)
interp_v_y = RegularGridInterpolator((r_vals, phi_vals), v_y, bounds_error=False, fill_value=0)
interp_pressure = RegularGridInterpolator((r_vals, phi_vals), pressure_vals, bounds_error=False, fill_value=P0)
interp_temperature = RegularGridInterpolator((r_vals, phi_vals), temperature_vals, bounds_error=False, fill_value=0)

# Evaluar los valores a lo largo de la trayectoria
points_traj = np.vstack((r_traj, phi_traj)).T  # Puntos en (r, φ)
v_r_traj = interp_v_r(points_traj)
v_phi_traj = interp_v_phi(points_traj)
v_y_traj = interp_v_y(points_traj)
pressure_traj = interp_pressure(points_traj)
temperature_traj = interp_temperature(points_traj)

# ===============================================================
# PLOT DE LA TRAYECTORIA Y LOS VALORES MEDIDOS (VELOCIDADES SEPARADAS)
# ===============================================================
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

# Velocidad radial (v_r)
axs[0].plot(x_traj, v_r_traj, color="blue")
axs[0].set_ylabel("$v_r$ (m/s)")
axs[0].grid(True)
axs[0].set_title("Velocidad radial a lo largo de la trayectoria del satélite")

# Velocidad azimutal (v_φ)
axs[1].plot(x_traj, v_phi_traj, color="orange")
axs[1].set_ylabel("$v_\\phi$ (m/s)")
axs[1].grid(True)
axs[1].set_title("Velocidad azimutal a lo largo de la trayectoria del satélite")

# Velocidad axial (v_y)
axs[2].plot(x_traj, v_y_traj, color="green")
axs[2].set_ylabel("$v_y$ (m/s)")
axs[2].grid(True)
axs[2].set_title("Velocidad axial a lo largo de la trayectoria del satélite")

# Presión
axs[3].plot(x_traj, pressure_traj, color="red")
axs[3].set_ylabel("Presión (Pa)")
axs[3].grid(True)
axs[3].set_title("Presión a lo largo de la trayectoria del satélite")

# Temperatura
axs[4].plot(x_traj, temperature_traj, color="purple")
axs[4].set_xlabel("Posición $x$ (m)")
axs[4].set_ylabel("Temperatura (K)")
axs[4].grid(True)
axs[4].set_title("Temperatura a lo largo de la trayectoria del satélite")

plt.tight_layout()

# ===============================================================
# PLOT MEJORADO DE LA TRAYECTORIA DEL SATÉLITE
# ===============================================================
fig_traj, ax_traj = plt.subplots(figsize=(10, 8))
cf = ax_traj.contourf(X, Z, pressure_vals, cmap='inferno', levels=50)
fig_traj.colorbar(cf, ax=ax_traj, label="Presión (Pa)")
ax_traj.plot(x_traj, z_traj, 'w-', lw=2, label="Trayectoria del satélite")
ax_traj.plot(x_traj[0], z_traj[0], 'go', markersize=10, label="Entrada")
ax_traj.plot(x_traj[-1], z_traj[-1], 'ro', markersize=10, label="Salida")
mid_idx = n_points // 2
dx = x_traj[mid_idx + 1] - x_traj[mid_idx]
dz = z_traj[mid_idx + 1] - z_traj[mid_idx]
ax_traj.arrow(x_traj[mid_idx], z_traj[mid_idx], dx, dz, head_width=0.1*a, head_length=0.2*a, 
              fc='white', ec='white', lw=1.5)
ax_traj.set_title("Trayectoria del satélite a través de la Flux Rope")
ax_traj.set_xlabel("$x = \\delta r \\cos(\\phi)$ (m)")
ax_traj.set_ylabel("$z = r \\sin(\\phi)$ (m)")
ax_traj.axis('equal')
ax_traj.legend(loc="upper right")
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = delta * a * np.cos(theta)
z_circle = a * np.sin(theta)
ax_traj.plot(x_circle, z_circle, 'b--', lw=1, label="Límite de la flux rope (r = a)")

plt.tight_layout()
plt.show()

# ===============================================================
# TRANSFORMACIÓN A COMPONENTES CARTESIANAS Vx Y Vz
# ===============================================================
# Convertir v_r y v_φ a Vx y Vz usando las relaciones cilíndricas
Vx_traj = v_r_traj * np.cos(phi_traj) - v_phi_traj * np.sin(phi_traj)
Vz_traj = v_r_traj * np.sin(phi_traj) + v_phi_traj * np.cos(phi_traj)

# Diagnóstico: Imprimir rangos de Vx y Vz para verificar variación
print("Rango de Vx_traj:", Vx_traj.min(), "a", Vx_traj.max())
print("Rango de Vz_traj:", Vz_traj.min(), "a", Vz_traj.max())

# ===============================================================
# PLOT SIMPLIFICADO DE Vx FRENTE A Vz
# ===============================================================
fig_vx_vz, ax_vx_vz = plt.subplots(figsize=(8, 6))
ax_vx_vz.plot(Vx_traj, Vz_traj, 'b-', lw=2, label="Trayectoria en el plano Vx-Vz")
ax_vx_vz.set_title("Velocidad en el plano Vx vs Vz a lo largo de la trayectoria")
ax_vx_vz.set_xlabel("$V_x$ (m/s)")
ax_vx_vz.set_ylabel("$V_z$ (m/s)")
ax_vx_vz.grid(True)
ax_vx_vz.legend(loc="best")
ax_vx_vz.axis('equal')  # Para mantener proporciones reales en el plano Vx-Vz

plt.tight_layout()
plt.show()