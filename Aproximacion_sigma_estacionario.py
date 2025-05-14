import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
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
mu0 = 4 * np.pi * 1e-7      # Vacuum permeability (H/m)
rho = 1e-20 # Mass density (kg/m³)
alpha = np.sqrt(1 / (rho * mu0))  # Factor α (m/s)
k_B = 1.380649e-23          # Boltzmann constant (J/K)
n = 1e7                    # Particle density (m⁻³) for temperature calculations
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
# T(r,φ) = P(r,φ) / (n * k_B)
# ===============================================================
temperature_vals = pressure_vals / (n * k_B)

# ===============================================================
# COMBINED PLOTS: PRESSURE AND TEMPERATURE
# ===============================================================
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Pressure subplot
cp0 = axs[0].contourf(X, Z, pressure_vals, cmap='plasma', levels=50)
cs0 = axs[0].contour(X, Z, pressure_vals, levels=10, colors='k', linewidths=0.5)
axs[0].clabel(cs0, inline=True, fontsize=8, fmt="%.1e")
axs[0].set_title("Pressure Distribution")
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$z$")
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
# SYNTHETIC TRAJECTORIES: Stagnation Pressure and Temperature
# ===============================================================
# Define synthetic trajectory function for P_stag and T_stag
from scipy.interpolate import RegularGridInterpolator

# ===============================================================
# VELOCITY FIELD CALCULATION
# ===============================================================
# It is assumed that B_r = 0, so v_r = 0.
v_r   = np.zeros_like(Bphi_vals)    # Radial component v_r
v_phi = alpha * Bphi_vals             # Azimuthal component v_φ
v_y   = alpha * By_vals               # Axial component v_y

# Prepare interpolators for velocity and field components
phi_grid = phi_vals
r_grid = r_vals

# Velocity field components (already computed previously)
# v_r, v_phi, v_y have shape (len(r_grid), len(phi_grid))
interp_v_r   = RegularGridInterpolator((r_grid, phi_grid), v_r,   bounds_error=False, fill_value=0)
interp_v_phi = RegularGridInterpolator((r_grid, phi_grid), v_phi, bounds_error=False, fill_value=0)
interp_v_y   = RegularGridInterpolator((r_grid, phi_grid), v_y,   bounds_error=False, fill_value=0)

# Pressure and temperature fields
interp_pressure    = RegularGridInterpolator((r_grid, phi_grid), pressure_vals,    bounds_error=False, fill_value=0)
interp_temperature = RegularGridInterpolator((r_grid, phi_grid), temperature_vals, bounds_error=False, fill_value=0)

# Physical constants for stagnation calculations
m_p = 1.6726219e-27  # Proton mass (kg)
# Specific heat ratio for monatomic ideal gas
gamma = 5.0/3.0

def synthetic_traj_P_T(z0, n_points=200):
    # Limit x-range to the ellipse cross‑section at height z0 to avoid out‑of‑bounds segments
    x_max = delta * a * np.sqrt(max(0.0, 1.0 - (z0 / a)**2))
    x_vals = np.linspace(-x_max, x_max, n_points)[1:-1]
    z_vals = np.full_like(x_vals, z0)
    # Cylindrical coords
    r_vals   = np.sqrt((x_vals / delta)**2 + z_vals**2)
    phi_vals = np.mod(np.arctan2(z_vals, x_vals / delta) + 2*np.pi, 2*np.pi)
    pts = np.vstack((r_vals, phi_vals)).T
    # interpolate fields
    v_r_t   = interp_v_r(pts)
    v_phi_t = interp_v_phi(pts)
    v_y_t   = interp_v_y(pts)
    # Solar wind background velocity for pressure
    v_sw = 400e3
    # Local velocity vector components
    v_local_x = v_phi_t * r_vals * np.cos(phi_vals)
    v_local_z = v_phi_t * r_vals * np.sin(phi_vals)
    v_local_y = v_y_t
    # Add solar wind vector (assumed along y-direction)
    v_total_x = v_local_x
    v_total_z = v_local_z+ v_sw
    v_total_y = v_local_y  # assuming solar wind in y-direction
    # Compute total velocity magnitude
    v_tot_pressure = np.sqrt(v_total_x**2 + v_total_y**2 + v_total_z**2)
    P_stag  = interp_pressure(pts) + 0.5 * rho * v_tot_pressure**2
    # Compute local sound speed and Mach number
    T_local = interp_temperature(pts)
    c_s = 25e-9/(np.sqrt(rho*mu0))
    T_stag = T_local* (1 + (v_tot_pressure / c_s)**2 / (gamma - 1))
    return x_vals, P_stag, T_stag

# Generate 10 trajectories
z0_vals = np.linspace(-0.9 * a, 0.9 * a, 10)
traj_data = [synthetic_traj_P_T(z0) for z0 in z0_vals]

# Plot P_stag and T_stag for each trajectory
fig_syn, (axP, axT) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for z0, (xv, Pv, Tv) in zip(z0_vals, traj_data):
    linestyle = '-' if z0 <= 0 else '--'
    color = cm.plasma((z0 + 0.9*a) / (1.8*a))
    axP.plot(xv, Pv, linestyle=linestyle, label=f"z={z0:.2e}", c=color)
    axT.plot(xv, Tv, linestyle=linestyle, label=f"z={z0:.2e}", c=color)

axP.set_ylabel("Stagnation Pressure (Pa)")
axP.set_title("Synthetic Trajectories: Stagnation Pressure")
axP.legend(fontsize="small", ncol=2, loc="upper right")
axP.grid(False)
axP.set_aspect('auto')

axT.set_xlabel("X position (m)")
axT.set_ylabel("Stagnation Temperature (K)")
axT.set_title("Synthetic Trajectories: Stagnation Temperature")
axT.legend(fontsize="small", ncol=2, loc="upper right")
axT.grid(False)
axT.set_aspect('auto')

plt.tight_layout()
plt.show()

# ===============================================================
# COMBINED PLOTS: VELOCITY COMPONENTS
# ===============================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

cp_vr = axs[0].contourf(X, Z, v_r, cmap='plasma', levels=50)
cs_vr = axs[0].contour(X, Z, v_r, levels=10, colors='k', linewidths=0.5)
axs[0].clabel(cs_vr, inline=True, fontsize=8, fmt="%.1e")
axs[0].set_title("$v_r$ (Radial Component)")
axs[0].set_xlabel("$x = \\delta r \\cos(\\phi)$")
axs[0].set_ylabel("$z = r \\sin(\\phi)$")
axs[0].axis("equal")
cbar = fig.colorbar(cp_vr, ax=axs[0], label="$v_r$")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)
cbar.ax.yaxis.get_offset_text().set_visible(True)
cbar.formatter.set_scientific(True)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()
axs[0].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

cp_vphi = axs[1].contourf(X, Z, v_phi*R_, cmap='plasma', levels=50)
cs_vphi = axs[1].contour(X, Z, v_phi*R_, levels=10, colors='k', linewidths=0.5)
axs[1].clabel(cs_vphi, inline=True, fontsize=8, fmt="%.1e")
axs[1].set_title("$v_\\phi$ (Azimuthal Component)")
axs[1].set_xlabel("$x$")
axs[1].set_ylabel("$z$")
axs[1].axis("equal")
cbar = fig.colorbar(cp_vphi, ax=axs[1], label="$v_\\phi$")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)
cbar.ax.yaxis.get_offset_text().set_visible(True)
cbar.formatter.set_scientific(True)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()
axs[1].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

cp_vy = axs[2].contourf(X, Z, v_y, cmap='plasma', levels=50)
cs_vy = axs[2].contour(X, Z, v_y, levels=10, colors='k', linewidths=0.5)
axs[2].clabel(cs_vy, inline=True, fontsize=8, fmt="%.1e")
axs[2].set_title("$v_y$ (Axial Component)")
axs[2].set_xlabel("$x$")
axs[2].set_ylabel("$z$")
axs[2].axis("equal")
cbar = fig.colorbar(cp_vy, ax=axs[2], label="$v_y$")
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)
cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar.ax.yaxis.offsetText.set_fontsize(10)
cbar.ax.yaxis.get_offset_text().set_visible(True)
cbar.formatter.set_scientific(True)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()
axs[2].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

plt.tight_layout()
plt.show()
