import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
n = 1e5                     # Particle density (m⁻³) for temperature calculations
P0 = 1e-12                   # Base pressure (Pa)

# ===============================================================
# MESH DEFINITION IN CYLINDRICAL COORDINATES
# ===============================================================
phi_vals = np.linspace(0, 2*np.pi, 200)
r_vals = np.linspace(0, a, 200)
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
# PRESSURE DISTRIBUTION CALCULATION
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

#
# ===============================================================
# TEMPERATURE DISTRIBUTION CALCULATION
# T(r,φ) = P(r,φ) / (2 * n * k_B)
# ===============================================================
# TEMPERATURE DISTRIBUTION CALCULATION
# T(r,φ) = P(r,φ) / (2 * n * k_B)
temperature_vals = pressure_vals / (2 * n * k_B)

# ===============================================================
# COMBINED PLOTS: PRESSURE AND TEMPERATURE
# ===============================================================
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Pressure subplot
cp0 = axs[0].contourf(X, Z, pressure_vals, cmap='inferno', levels=50)
cs0 = axs[0].contour(X, Z, pressure_vals, levels=10, colors='k', linewidths=0.5)
axs[0].clabel(cs0, inline=True, fontsize=8, fmt="%.1e")
axs[0].set_title("Pressure Distribution")
axs[0].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[0].set_ylabel("$z = r \\sin(\\phi)$")
axs[0].axis('equal')
cbar = fig.colorbar(cp0, ax=axs[0], label="Pressure (Pa)")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)

# Temperature subplot
cp1 = axs[1].contourf(X, Z, temperature_vals, cmap='plasma', levels=50)
cs1 = axs[1].contour(X, Z, temperature_vals, levels=10, colors='k', linewidths=0.5)
axs[1].clabel(cs1, inline=True, fontsize=8, fmt="%.1e")
axs[1].set_title("Temperature Distribution")
axs[1].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[1].set_ylabel("$z = r \\sin(\\phi)$")
axs[1].axis('equal')
cbar = fig.colorbar(cp1, ax=axs[1], label="Temperature (K)")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)

plt.tight_layout()
plt.show()

# ===============================================================
# VELOCITY FIELD CALCULATION
# ===============================================================
# It is assumed that B_r = 0, so v_r = 0.
v_r   = np.zeros_like(Bphi_vals)    # Radial component v_r
v_phi = alpha * Bphi_vals             # Azimuthal component v_φ
v_y   = alpha * By_vals               # Axial component v_y

# ===============================================================
# COMBINED PLOTS: VELOCITY COMPONENTS
# ===============================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

cp_vr = axs[0].contourf(X, Z, v_r, cmap='viridis', levels=50)
cs_vr = axs[0].contour(X, Z, v_r, levels=10, colors='k', linewidths=0.5)
axs[0].clabel(cs_vr, inline=True, fontsize=8, fmt="%.1e")
axs[0].set_title("$v_r$ (Radial Component)")
axs[0].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[0].set_ylabel("$z = r \\sin(\\phi)$")
axs[0].axis("equal")
cbar = fig.colorbar(cp_vr, ax=axs[0], label="$v_r$")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)

cp_vphi = axs[1].contourf(X, Z, v_phi, cmap='plasma', levels=50)
cs_vphi = axs[1].contour(X, Z, v_phi, levels=10, colors='k', linewidths=0.5)
axs[1].clabel(cs_vphi, inline=True, fontsize=8, fmt="%.1e")
axs[1].set_title("$v_\\phi$ (Azimuthal Component)")
axs[1].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[1].set_ylabel("$z = r \\sin(\\phi)$")
axs[1].axis("equal")
cbar = fig.colorbar(cp_vphi, ax=axs[1], label="$v_\\phi$")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)

cp_vy = axs[2].contourf(X, Z, v_y, cmap='cividis', levels=50)
cs_vy = axs[2].contour(X, Z, v_y, levels=10, colors='k', linewidths=0.5)
axs[2].clabel(cs_vy, inline=True, fontsize=8, fmt="%.1e")
axs[2].set_title("$v_y$ (Axial Component)")
axs[2].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[2].set_ylabel("$z = r \\sin(\\phi)$")
axs[2].axis("equal")
cbar = fig.colorbar(cp_vy, ax=axs[2], label="$v_y$")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)

plt.tight_layout()
plt.show()

# ===============================================================
# SYNTHETIC ANALYSIS: SATELLITE TRAJECTORY
# ===============================================================
from scipy.interpolate import RegularGridInterpolator

# Define the satellite trajectory in the (x, z) plane
x_start, x_end = -delta * a, delta * a  # Entry and exit points
z_fixed = 0.0                           # Set z = 0 (can be changed)
n_points = 100                          # Number of points in the trajectory
x_traj = np.linspace(x_start, x_end, n_points)  # x coordinates
z_traj = np.zeros_like(x_traj)                  # z coordinates (constant)

# Convert the trajectory (x, z) to cylindrical coordinates (r, φ)
r_traj = np.sqrt((x_traj / delta)**2 + z_traj**2)  # r = sqrt((x/δ)² + z²)
phi_traj = np.arctan2(z_traj, x_traj / delta)      # φ = arctan(z / (x/δ))
phi_traj = np.mod(phi_traj + 2 * np.pi, 2 * np.pi)

# Create interpolators for the calculated fields
interp_v_r = RegularGridInterpolator((r_vals, phi_vals), v_r, bounds_error=False, fill_value=0)
interp_v_phi = RegularGridInterpolator((r_vals, phi_vals), v_phi, bounds_error=False, fill_value=0)
interp_v_y = RegularGridInterpolator((r_vals, phi_vals), v_y, bounds_error=False, fill_value=0)
interp_pressure = RegularGridInterpolator((r_vals, phi_vals), pressure_vals, bounds_error=False, fill_value=P0)
interp_temperature = RegularGridInterpolator((r_vals, phi_vals), temperature_vals, bounds_error=False, fill_value=0)

# Evaluate values along the trajectory
points_traj = np.vstack((r_traj, phi_traj)).T  # Points in (r, φ)
v_r_traj = interp_v_r(points_traj)
v_phi_traj = interp_v_phi(points_traj)
v_y_traj = interp_v_y(points_traj)
pressure_traj = interp_pressure(points_traj)
temperature_traj = interp_temperature(points_traj)

#
# Calculate velocity magnitude along the trajectory using local cylindrical radius
# v_total = sqrt(v_r^2 + (v_phi * r_local)^2 + v_y^2)
r_local_traj = r_traj  # Since r_traj already represents the local radius
v_total_traj = np.sqrt(v_r_traj**2 + (v_phi_traj * r_local_traj)**2 + v_y_traj**2)

# Stagnation Pressure: P_total = P + 0.5 * rho * v^2
pressure_stag_traj = pressure_traj + 0.5 * rho * v_total_traj**2

# Stagnation Temperature: T_total = T + (m / 2kB) * v^2
m_p = 1.6726219e-27  # Proton mass (kg) as approximation for plasma particles
temperature_stag_traj = temperature_traj + (m_p / (2 * k_B)) * v_total_traj**2

# ===============================================================
# PLOT OF THE TRAJECTORY AND MEASURED VALUES
# ===============================================================
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

# Radial velocity (v_r)
axs[0].plot(x_traj, v_r_traj, color="blue")
axs[0].set_ylabel("$v_r$ (m/s)")
axs[0].grid(True)
axs[0].set_title("Radial velocity along the satellite trajectory")

# Azimuthal velocity (v_φ)
axs[1].plot(x_traj, v_phi_traj, color="orange")
axs[1].set_ylabel("$v_\\phi$ (m/s)")
axs[1].grid(True)
axs[1].set_title("Azimuthal velocity along the satellite trajectory")

# Axial velocity (v_y)
axs[2].plot(x_traj, v_y_traj, color="green")
axs[2].set_ylabel("$v_y$ (m/s)")
axs[2].grid(True)
axs[2].set_title("Axial velocity along the satellite trajectory")

# Stagnation Pressure
axs[3].plot(x_traj, pressure_stag_traj, color="red")
axs[3].set_ylabel("Stagnation Pressure (Pa)")
axs[3].grid(True)
axs[3].set_title("Stagnation Pressure along the satellite trajectory")

# Stagnation Temperature
axs[4].plot(x_traj, temperature_stag_traj, color="purple")
axs[4].set_xlabel("Position $x$ (m)")
axs[4].set_ylabel("Stagnation Temperature (K)")
axs[4].grid(True)
axs[4].set_title("Stagnation Temperature along the satellite trajectory")

plt.tight_layout()

# ===============================================================
# ENHANCED PLOT OF THE SATELLITE TRAJECTORY
# ===============================================================
fig_traj, ax_traj = plt.subplots(figsize=(10, 8))
cf = ax_traj.contourf(X, Z, pressure_vals, cmap='inferno', levels=50)
cbar = fig_traj.colorbar(cf, ax=ax_traj, label="Pressure (Pa)")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)
ax_traj.plot(x_traj, z_traj, 'w-', lw=2, label="Satellite trajectory")
ax_traj.plot(x_traj[0], z_traj[0], 'go', markersize=10, label="Entry")
ax_traj.plot(x_traj[-1], z_traj[-1], 'ro', markersize=10, label="Exit")
mid_idx = n_points // 2
dx = x_traj[mid_idx + 1] - x_traj[mid_idx]
dz = z_traj[mid_idx + 1] - z_traj[mid_idx]
ax_traj.arrow(x_traj[mid_idx], z_traj[mid_idx], dx, dz, head_width=0.1*a, head_length=0.2*a, 
              fc='white', ec='white', lw=1.5)
ax_traj.set_title("Satellite trajectory through the Flux Rope")
ax_traj.set_xlabel("$x = \\delta r \\cos(\\phi)$ (m)")
ax_traj.set_ylabel("$z = r \\sin(\\phi)$ (m)")
ax_traj.axis('equal')
ax_traj.legend(loc="upper right")
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = delta * a * np.cos(theta)
z_circle = a * np.sin(theta)
ax_traj.plot(x_circle, z_circle, 'b--', lw=1, label="Flux rope boundary (r = a)")

plt.tight_layout()
plt.show()

# ===============================================================
# TRANSFORMATION TO CARTESIAN COMPONENTS Vx AND Vz
# ===============================================================
# Convert v_r and v_φ to Vx and Vz using cylindrical relations
Vx_traj = v_r_traj * np.cos(phi_traj) - v_phi_traj * np.sin(phi_traj)
Vz_traj = v_r_traj * np.sin(phi_traj) + v_phi_traj * np.cos(phi_traj)

# Diagnostic: Print Vx and Vz ranges to verify variation
print("Rango de Vx_traj:", Vx_traj.min(), "a", Vx_traj.max())
print("Rango de Vz_traj:", Vz_traj.min(), "a", Vz_traj.max())

# ===============================================================
# SIMPLIFIED PLOT OF Vx VS Vz
# ===============================================================
fig_vx_vz, ax_vx_vz = plt.subplots(figsize=(8, 6))
ax_vx_vz.plot(Vx_traj, Vz_traj, 'b-', lw=2, label="Trajectory in the Vx-Vz plane")
ax_vx_vz.set_title("Velocity in the Vx vs Vz plane along the trajectory")
ax_vx_vz.set_xlabel("$V_x$ (m/s)")
ax_vx_vz.set_ylabel("$V_z$ (m/s)")
ax_vx_vz.grid(True)
ax_vx_vz.set_xscale('log')
ax_vx_vz.set_yscale('log')
ax_vx_vz.legend(loc="best")
ax_vx_vz.axis('equal')  # To maintain real proportions in the Vx-Vz plane

plt.tight_layout()
plt.show()