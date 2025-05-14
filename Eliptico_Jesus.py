import sympy as sp
import numpy as np
from scipy.integrate import quad, dblquad, cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Nuevo import para colormaps “inferno”
import matplotlib.ticker as ticker

import pandas as pd

# --- Configuración de Modo ---
modo_rapido = True
computar_helicidad = False  # Cambia a True si deseas calcular la helicidad

plt.rcParams.update({'font.size': 12})
plt.rcParams.update({
    "text.usetex": not modo_rapido,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
})

#######################################
### Definir variables simbólicas ######
#######################################
r, phi, u = sp.symbols('r phi u', real=True)

#######################################
### Constantes físicas del sistema ####
#######################################
mu_0 = 4 * sp.pi * 1e-7  
# Numeric constant for mu_0
mu0_num = 4 * np.pi * 1e-7
R = 0.7e10    # Semieje menor
a = 1e10    # Semieje mayor
L = 10e10     # Longitud del cilindro
                                                                 #OK
#######################################
### Parámetros de ajuste para j #########
#######################################

alfa = 1e-32
beta = 1e-22
delta = R/a
By0 = mu_0*delta*alfa*a**3/3 # Campo magnético axial en el origen

#######################################
### Definición de la métrica ############
#######################################
g_rr = delta**2 * sp.cos(phi)**2 + sp.sin(phi)**2
g_yy = 1
g_phiphi = r**2 * (delta**2 * sp.sin(phi)**2 + sp.cos(phi)**2)
g_rphi = r * (1 - delta**2) * sp.sin(phi) * sp.cos(phi)

#######################################
### Chi  ###############################
#######################################
h_sq = delta**2 * sp.sin(phi)**2 + sp.cos(phi)**2
chi = (delta**2 + 1) / h_sq

#######################################
### Corrientes ###########################
#######################################
j_phi_phi = (
    0.12615662610100803
    - 0.24000778968602718 * sp.cos(1 * phi)
    + 0.20657661898691135 * sp.cos(2 * phi)
    - 0.16088203263124973 * sp.cos(3 * phi)
    + 0.11337165224497914 * sp.cos(4 * phi)
    - 0.07228895706727245 * sp.cos(5 * phi)
    + 0.041707100072565964 * sp.cos(6 * phi)
    - 0.0217730154538321 * sp.cos(7 * phi)
    + 0.010284844252703521 * sp.cos(8 * phi)
    - 0.004395896006372525 * sp.cos(9 * phi)
    + 0.0017000733205040617 * sp.cos(10 * phi)
    - 0.0005949198311010654 * sp.cos(11 * phi)
    + 0.0001883734933593818 * sp.cos(12 * phi)
)

j_phi_r = -alfa*r
j_y_r = beta*r

#######################################
### Cálculo de jr ######################
#######################################
dj_phi_dphi = sp.diff(j_phi_phi, phi)
rp = sp.Symbol("rp", real=True)
integral_j_phi_r_expr = sp.integrate(rp * j_phi_r.subs(r, rp), (rp, 0, r))
jr = - (dj_phi_dphi / r) * integral_j_phi_r_expr

#######################################
### Cálculo de jphi ####################
#######################################
jphi = j_phi_phi * j_phi_r

#######################################
### Campos magnéticos antiguos #########
#######################################
Br = 0
Bphi_old = - (mu_0 * delta / (g_phiphi * r**(chi - 2))) * sp.integrate(r**(chi - 1) * j_y_r, (r, 0, r))
By = By0 + mu_0 * delta * j_phi_phi * sp.integrate(r * j_phi_r, (r, 0, r))

#######################################
### Flujos magnéticos ##################
#######################################
By_func = sp.lambdify((r, phi), By, "numpy")
g_rr_func = sp.lambdify(phi, g_rr, "numpy")
g_phiphi_func = sp.lambdify((r, phi), g_phiphi, "numpy")
g_rphi_func = sp.lambdify((r, phi), g_rphi, "numpy")

integrand = lambda phi, r: By_func(r, phi) * delta * r
flujo_axial, error = dblquad(integrand, 0, R, lambda phi: 0, lambda phi: 2 * np.pi)
print("Flujo Axial Numérico:", flujo_axial)

flujo_poloidal = sp.integrate(Bphi_old * delta * r, (r, 0, R))

#######################################
### Corriente axial poloidal ###########
#######################################

Bphi_old_cc = - (mu_0 * delta / r**2) * sp.integrate(r * j_y_r, (r, 0, r))
# Convertir la expresión simbólica en una función numérica:
Bphi_old_cc_func = sp.lambdify(r, Bphi_old_cc, "numpy")
integrand_cc = lambda r: Bphi_old_cc_func(r) * r
flujo_cc, error = quad(integrand_cc, 0, R)
j_y_phi = (flujo_cc / flujo_poloidal)

#######################################
### Cálculo de jy ######################
#######################################
jy = j_y_phi * j_y_r

#######################################
### Cálculo de Bphi ####################
#######################################
Bphi = Bphi_old * j_y_phi

#######################################
### Plot de campos y corrientes ##########
#######################################
Bphi_func = sp.lambdify((r, phi), Bphi, "numpy")
By_func = sp.lambdify((r, phi), By, "numpy")
jr_func = sp.lambdify((r, phi), jr, "numpy")
jphi_func = sp.lambdify((r, phi), jphi, "numpy")
jy_func = sp.lambdify((r, phi), jy, "numpy")

epsilon = 1e-3
if modo_rapido:
    phi_vals = np.linspace(0, 2 * np.pi, 50)
    r_vals   = np.linspace(epsilon, a, 50)
else:
    phi_vals = np.linspace(0, 2 * np.pi, 100)
    r_vals   = np.linspace(epsilon, a, 100)
Phi, R_ = np.meshgrid(phi_vals, r_vals)
X = delta * R_ * np.cos(Phi)
Z = R_ * np.sin(Phi)

factor=R_*((delta**2 * np.sin(Phi)**2 + np.cos(Phi)**2))**0.5

Bphi_vals = Bphi_func(R_, Phi)
By_vals = By_func(R_, Phi)
jr_vals_raw = jr_func(R_, Phi)
if np.ndim(jr_vals_raw) == 0:
    jr_vals = np.full_like(R_, jr_vals_raw)
else:
    jr_vals = jr_vals_raw
jphi_vals = jphi_func(R_, Phi)
jy_vals = jy_func(R_, Phi)

def plot_contour(ax, X, Z, values, title):
    # Colormap “plasma” (más suave), rejilla desactivada
    levels_f = 30 if modo_rapido else 50
    levels_c = 5 if modo_rapido else 7
    c = ax.contourf(X, Z, values, cmap="plasma", levels=levels_f)
    fig_tmp = ax.get_figure()
    cbar = fig_tmp.colorbar(c, ax=ax)
    cbar.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.offsetText.set_fontsize(10)
    # Añadir curvas de nivel (líneas de contorno)
    lines = ax.contour(X, Z, values, levels=levels_c, colors='k', linewidths=1)
    has_contours = any(len(level) > 0 for level in lines.allsegs)
    if has_contours:
        try:
            ax.clabel(lines, inline=True, fontsize=8, fmt="%.1e", inline_spacing=5)
        except IndexError:
            print(f"⚠️  No se pudieron etiquetar las curvas en: {title}")
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
# --- FIGURA 1: Campos Magnéticos ---
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
fig1.tight_layout()
plot_contour(axes1[0], X, Z, By_vals, "$B^y$ (Magnetic Field)")
plot_contour(axes1[1], X, Z, Bphi_vals*factor, "$B^\\varphi$ (Magnetic Field)")
plt.show()
fig1.savefig("fig1.svg", format="svg")

# --- FIGURA 2: Corrientes ---------------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.tight_layout()
plot_contour(axes2[0], X, Z, jr_vals, "$j^r$ (Current Field)")
plot_contour(axes2[1], X, Z, jy_vals, "$j^y$ (Current Field)")
plot_contour(axes2[2], X, Z, jphi_vals*factor, "$j^\\phi$ (Current Field)")
plt.show()
fig2.savefig("fig2.svg", format="svg")

# --- FIGURA 3: Fuerzas de Lorentz (optimizada) ---
# Redefinir malla con resolución reducida si modo_rapido
if modo_rapido:
    n_phi3, n_r3 = 50, 50
else:
    n_phi3, n_r3 = len(phi_vals), len(r_vals)
phi_vals3 = np.linspace(0, 2 * np.pi, n_phi3)
r_vals3 = np.linspace(epsilon, a, n_r3)
Phi3, R3 = np.meshgrid(phi_vals3, r_vals3)
X3 = delta * R3 * np.cos(Phi3)
Z3 = R3 * np.sin(Phi3)

# Compute metric factors on the R3, Phi3 mesh
g_rr_vals3     = g_rr_func(Phi3)
g_phiphi_vals3 = g_phiphi_func(R3, Phi3)
g_rphi_vals3   = g_rphi_func(R3, Phi3)

# Extract current and field arrays on the R3, Phi3 mesh
jr_vals3   = jr_func(R3, Phi3)
jphi_vals3 = jphi_func(R3, Phi3)
jy_vals3   = jy_func(R3, Phi3)
Bphi_vals3 = Bphi_func(R3, Phi3)
By_vals3   = By_func(R3, Phi3)

# Lorentz force components using metric
f_r_vals = (g_yy * g_phiphi_vals3) / (delta * R3) * (jy_vals3 * Bphi_vals3 - jphi_vals3 * By_vals3)
f_y_vals = (1 / (delta * R3)) * (
    (g_phiphi_vals3 * jphi_vals3 + g_rphi_vals3 * jr_vals3) * g_rphi_vals3 * Bphi_vals3
    - (g_rphi_vals3   * jphi_vals3 + g_rr_vals3   * jr_vals3) * g_phiphi_vals3 * Bphi_vals3
)
f_phi_vals = (g_rr_vals3 * g_yy) / (delta * R3) * jr_vals3 * By_vals3

factor3 = R3 * ((delta**2 * np.sin(Phi3)**2 + np.cos(Phi3)**2))**0.5

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.tight_layout()
plot_contour(axes3[0], X3, Z3, f_r_vals, "$f^r$ (Lorentz Force)")
plot_contour(axes3[1], X3, Z3, f_y_vals, "$f^y$ (Lorentz Force)")
plot_contour(axes3[2], X3, Z3, f_phi_vals * factor3, "$f^\\phi$ (Lorentz Force)")
plt.show()
fig3.savefig("fig3.svg", format="svg")


# --- FIGURA 4: Corriente Perpendicular y Desalineamiento/Energía -----------
# Ensure j_r, j_y, j_phi, B_r, B_y, B_phi are defined just before this block
j_r = jr_vals
j_y = jy_vals
j_phi = jphi_vals
B_r = np.zeros_like(j_r)
B_y = By_vals
B_phi = Bphi_vals

# Load metric factors on the same mesh
g_rr_vals     = g_rr_func(Phi)
g_phiphi_vals = g_phiphi_func(R_, Phi)
g_yy_val      = 1

# Magnitudes with metric
mag_jxB = np.sqrt(g_rr_vals * f_r_vals**2
                + g_yy_val * f_y_vals**2
                + g_phiphi_vals * f_phi_vals**2)
mag_j   = np.sqrt(g_rr_vals * j_r**2
                + g_yy_val * j_y**2
                + g_phiphi_vals * j_phi**2)
mag_B   = np.sqrt(g_rr_vals * B_r**2
                + g_yy_val * B_y**2
                + g_phiphi_vals * B_phi**2)

# Misalignment factor
sin_omega = mag_jxB / (mag_j * mag_B)

jperp = (g_yy * g_phiphi / (delta * r * sp.sqrt(g_yy*By*By + g_phiphi*Bphi*Bphi))) * (jy * Bphi - jphi * By)
jperp = sp.simplify(jperp)
j_perp_func = sp.lambdify((r, phi), jperp, "numpy")
j_perp_raw = j_perp_func(R_, Phi)
if np.ndim(j_perp_raw) == 0:
    j_perp_values = np.full_like(R_, j_perp_raw)
else:
    j_perp_values = j_perp_raw

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
fig4.tight_layout()
plot_contour(axes4[0], X, Z, j_perp_values, "$j_{\\perp}$ (Perpendicular Current)")
plot_contour(axes4[1], X, Z, sin_omega, "$sin(\\omega)$")

# Compute magnetic energy density numerically
gphiphi_vals = g_phiphi_func(R_, Phi)
magE_vals = (By_vals**2 + gphiphi_vals * Bphi_vals**2) / (2 * mu0_num)
plot_contour(axes4[2], X, Z, magE_vals, "Magnetic Energy Density")
plt.show()
fig4.savefig("fig4.svg", format="svg")

#######################################
############## Helicidad ##############
#######################################

def Ay_numeric(r_val, phi_val):
    integrand = lambda rp: rp * Bphi_func(rp, phi_val)
    result, err = quad(integrand, 0, r_val)
    return delta * result

def D_numeric(r_val, phi_val):
    def inner_integral(rp):
        lower = phi_val - (r_val - rp)
        upper = phi_val + (r_val - rp)
        if upper > lower:
            result, err = quad(lambda phip: By_func(rp, phip), lower, upper, limit=200, epsabs=1e-6, epsrel=1e-6)
            return result
        else:
            return 0
    result_outer, err = quad(lambda rp: rp * inner_integral(rp), 0, r_val, limit=200)
    return (delta / 2) * result_outer

def compute_derivatives_D(r_val, phi_val, h=1e-5):
    dD_dr = (D_numeric(r_val + h, phi_val) - D_numeric(r_val - h, phi_val)) / (2 * h)
    dD_dphi = (D_numeric(r_val, phi_val + h) - D_numeric(r_val, phi_val - h)) / (2 * h)
    return dD_dr, dD_dphi

def Ar_numeric(r_val, phi_val):
    dD_dr, dD_dphi = compute_derivatives_D(r_val, phi_val)
    return (g_phiphi_func(r_val, phi_val) * dD_dphi - g_rphi_func(r_val, phi_val) * dD_dr) / (delta * r_val)

def Aphi_numeric(r_val, phi_val):
    dD_dr, dD_dphi = compute_derivatives_D(r_val, phi_val)
    return (g_rr_func(phi_val) * dD_dr - g_rphi_func(r_val, phi_val) * dD_dphi) / (delta * r_val)

def helicity_integrand(r_val, phi_val):
    Ay_val    = Ay_numeric(r_val, phi_val)
    Ar_val    = Ar_numeric(r_val, phi_val)
    Aphi_val  = Aphi_numeric(r_val, phi_val)
    Bphi_val  = Bphi_func(r_val, phi_val)
    By_val    = By_func(r_val, phi_val)
    return ( g_rphi_func(r_val, phi_val) * Ar_val * Bphi_val +
             Ay_val * By_val +
             g_phiphi_func(r_val, phi_val) * Aphi_val * Bphi_val ) * delta * r_val * L

def integrand_for_dblquad(phi_val, r_val):
    return helicity_integrand(r_val, phi_val)

if computar_helicidad:
    H_numeric, err = dblquad(integrand_for_dblquad,
                             0, R,
                             lambda r_val: 0,
                             lambda r_val: 2*np.pi)
    print("Helicidad Magnética: ", H_numeric)
else:
    H_numeric = 0
    print("⚡ Helicity computation skipped (computar_helicidad = False)")

# --- Generar Reporte Completo de Magnitudes ---

from scipy.integrate import dblquad

mag_energy_total = dblquad(
    lambda phi, r: ((By_func(r, phi)**2 + g_phiphi_func(r, phi) * Bphi_func(r, phi)**2)
                    / (2 * mu0_num)) * delta * r,
    0, R,
    lambda r: 0,
    lambda r: 2 * np.pi
)[0]

report_data = {
    "Parametro": [
        "Flujo Axial Numérico",
        "Corriente Axial Poloidal j_y_phi",
        "Helicidad Magnética",
        "Energía Magnética Total",
        "Beta",
        "Alfa",
        "Delta",
        "By0",
        "Semieje menor R",
        "Semieje mayor a",
        "Longitud L"
    ],
    "Valor": [
        flujo_axial,
        j_y_phi,
        H_numeric,
        mag_energy_total,
        beta,
        alfa,
        delta,
        By0,
        R,
        a,
        L
    ]
}

# Crear DataFrame y mostrar
df_report = pd.DataFrame(report_data)
print("\n--- Reporte Completo de Magnitudes ---")
print(df_report)

# Exportar a CSV
df_report.to_csv("reporte_completo_magnitudes.csv", index=False)
print("✅ Reporte guardado en 'reporte_completo_magnitudes.csv'")
print("✅ Expresiones analíticas guardadas en 'expresiones_analiticas.tex'")

# --- Exportar Expresiones Analíticas Simplificadas a LaTeX ---
expresiones = {
    r"\textbf{B}^\phi(r, \phi)": sp.simplify(Bphi),
    r"\textbf{B}^y(r, \phi)": sp.simplify(By),
    r"\textbf{j}^r(r, \phi)": sp.simplify(jr),
    r"\textbf{j}^\phi(r, \phi)": sp.simplify(jphi),
    r"\textbf{j}^y(r, \phi)": sp.simplify(jy)
}

with open("expresiones_analiticas.tex", "w") as f:
    f.write(r"\section*{Expresiones Analíticas del Modelo}" + "\n\n")
    f.write(r"% Expresiones principales del modelo: Campos magnéticos y corrientes" + "\n\n")
    for nombre, expr in expresiones.items():
        f.write(r"$" + nombre + r" = " + sp.latex(expr) + r"$" + "\n\n")


# --- FIGURA CURVA PARAMÉTRICA OPTIMIZADA ---
print("🎨 Graficando conjunto de curvas paramétricas (vectorizado)...")

vueltas = 5
# Resolución reducida para prototipado
n_phi = 500
n_radios = 10
phi_vals_curve = np.linspace(0, 2 * np.pi * vueltas, n_phi)
radios = np.linspace(0.1 * R, 0.9 * R, n_radios)

# Vectorizar cálculo
Phi2d, R2d = np.meshgrid(phi_vals_curve, radios)
By2d = By_func(R2d, Phi2d)
Bphi2d = Bphi_func(R2d, Phi2d)
# Evitar divisiones por cero
Bphi2d = np.where(np.abs(Bphi2d) < 1e-12, 1e-12, Bphi2d)
integrand2d = (By2d / Bphi2d) * R2d
# Integral acumulada por filas (eje φ)
Y2d = cumulative_trapezoid(integrand2d, phi_vals_curve, axis=1, initial=0)
X2d = delta * R2d * np.cos(Phi2d)
Z2d = R2d * np.sin(Phi2d)

# Change line colors and scale all curves to full cylinder length
max_y = np.max(Y2d[:, -1])

# Graficar cada curva, ya con datos precomputados
fig_curve = plt.figure(figsize=(12, 8))
ax_curve = fig_curve.add_subplot(111, projection='3d')
for i in range(n_radios):
    color = cm.viridis(i / (n_radios - 1))
    y_scaled = Y2d[i] * (max_y / Y2d[i, -1])
    ax_curve.plot(X2d[i], y_scaled, Z2d[i], lw=2.2, color=color)


# Añadir cilindro transparente
theta_cyl = np.linspace(0, 2 * np.pi, 100)
y_cyl = np.linspace(np.min(Y2d), np.max(Y2d), 100)
theta_grid, y_grid = np.meshgrid(theta_cyl, y_cyl)
r_cyl = 0.95 * R
x_cyl = delta * r_cyl * np.cos(theta_grid)
z_cyl = r_cyl * np.sin(theta_grid)
ax_curve.plot_surface(x_cyl, y_grid, z_cyl, color='lightgray', alpha=0.1, linewidth=0, antialiased=False)

    # Opcional: proyecciones
    # ax_curve.plot(X2d[i], Y2d[i], zs=0, zdir='z', linestyle='dotted', color='gray', alpha=0.3)
    # ax_curve.plot(X2d[i], np.zeros_like(Y2d[i]), Z2d[i], zdir='y', linestyle='dotted', color='gray', alpha=0.3)
    # ax_curve.plot(np.zeros_like(X2d[i]), Y2d[i], Z2d[i], zdir='x', linestyle='dotted', color='gray', alpha=0.3)

ax_curve.set_xlabel(r"$x$")
ax_curve.set_ylabel(r"$y = \int_0^\varphi \frac{B^y}{B^\varphi} r d\varphi$")
ax_curve.set_zlabel(r"$z")
ax_curve.set_title("Magnetic Field Lines (Parametric Curves)")
ax_curve.view_init(elev=25, azim=-60)  # Vista rotada
ax_curve.grid(False)
ax_curve.set_box_aspect([2, 4, 2])  # Aspecto personalizado
ax_curve.set_xticks([])
ax_curve.set_yticks([])
ax_curve.set_zticks([])
ax_curve.set_xlabel("")
ax_curve.set_ylabel("")
ax_curve.set_zlabel("")
# Remove the surrounding 3D axes box
ax_curve.set_axis_off()
plt.tight_layout()
plt.show()
fig_curve.savefig("fig_curve.svg", format="svg")
print("✅ Conjunto de curvas paramétricas graficado.")


# --- FIGURA ÁNGULO DE PITCH ---
print("📐 Graficando ángulo de pitch para distintos radios...")

fig_pitch, ax_pitch = plt.subplots(figsize=(10, 6))
radios_pitch = np.linspace(0.1 * R, 0.9 * R, 10)

for i, r_val in enumerate(radios_pitch):
    r_array = np.full_like(phi_vals_curve, r_val)
    By_vals = By_func(r_array, phi_vals_curve)
    Bphi_vals = Bphi_func(r_array, phi_vals_curve)
    Bphi_vals = np.where(np.abs(Bphi_vals) < 1e-12, 1e-12, Bphi_vals)
    alpha_vals = np.arctan(By_vals / Bphi_vals)  # Ángulo de pitch
    color = cm.plasma(i / (len(radios_pitch) - 1))
    ax_pitch.plot(phi_vals_curve, np.degrees(alpha_vals), label=f"r = {round(r_val/1e10,2)}e10", color=color)

ax_pitch.set_title("Pitch Angle vs Azimuthal Angle")
ax_pitch.set_xlabel(r"$\varphi$ (rad)")
ax_pitch.set_ylabel(r"$\alpha(r,\varphi)$ (degrees)")
ax_pitch.grid(True)
ax_pitch.legend()
plt.tight_layout()
plt.show()
fig_pitch.savefig("fig_pitch.svg", format="svg")
print("✅ Ángulo de pitch graficado.")


# --- Synthetic trajectory generation for constant z ---
def synthetic_trajectory(z0, n_points=200):
    # Numeric parameters from symbolic constants
    delta_val = float(delta)
    a_val = float(a)
    R_val = float(R)
    # Compute x range within ellipse cross-section at height z0
    x_max = R_val * np.sqrt(max(0, 1 - (z0 / a_val)**2))
    x_vals = np.linspace(-x_max, x_max, n_points)
    # Convert (x, z0) into (r, phi) for evaluation
    r_vals = np.sqrt((x_vals / delta_val)**2 + z0**2)
    phi_vals = np.arctan2(z0, x_vals / delta_val)
    # Evaluate magnetic field components
    Bphi_vals = Bphi_func(r_vals, phi_vals)
    By_vals = By_func(r_vals, phi_vals)
    # Compute physical magnitude of Bphi
    factor_vals = r_vals * np.sqrt(delta_val**2 * np.sin(phi_vals)**2 + np.cos(phi_vals)**2)
    Bphi_phys = Bphi_vals * factor_vals
    # Total magnetic field magnitude
    Bmag = np.sqrt(Bphi_phys**2 + By_vals**2)
    return x_vals, Bmag


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec

    # Generate 10 synthetic trajectories at evenly spaced z positions within the ellipse
    z0_vals = np.linspace(-0.9 * float(a), 0.9 * float(a), 10)
    trajectories = [synthetic_trajectory(z0) for z0 in z0_vals]

    # --- Top-left: B^φ vs B^y for each trajectory ---
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 2],
                          hspace=0.3, wspace=0.3)
    ax_bvsb = fig.add_subplot(gs[0, 0])
    ax1     = fig.add_subplot(gs[0, 1])
    ax2     = fig.add_subplot(gs[1, :])

    # Plot B^φ vs B^y for each trajectory with unique color
    for i, (z0, (x_vals, _)) in enumerate(zip(z0_vals, trajectories)):
        # Recompute radial and angular positions
        delta_val = float(delta)
        r_vals = np.sqrt((x_vals / delta_val)**2 + z0**2)
        phi_vals = np.arctan2(z0, x_vals / delta_val)
        # Field components
        Bphi_vals = Bphi_func(r_vals, phi_vals)
        By_vals = By_func(r_vals, phi_vals)
        # Physical Bφ
        factor_vals = r_vals * np.sqrt(delta_val**2 * np.sin(phi_vals)**2 + np.cos(phi_vals)**2)
        Bphi_phys = Bphi_vals * factor_vals
        linestyle = '--' if z0 > 0 else '-'
        color = cm.plasma(i / (len(z0_vals) - 1))
        ax_bvsb.plot(Bphi_phys, By_vals, linestyle=linestyle, color=color, label=f"z = {z0:.2e}")
    ax_bvsb.set_xlabel(r"$B^\varphi$ (T)")
    ax_bvsb.set_ylabel(r"$B^y$ (T)")
    ax_bvsb.set_title(r"$B^\varphi$ vs $B^y$ along trajectories")
    ax_bvsb.legend(fontsize="small", loc="best")

    # --- Top-right: contour of field magnitude in X-Z plane ---
    # Prepare mesh
    phi_mesh = np.linspace(0, 2 * np.pi, 100)
    r_mesh = np.linspace(1e-3, float(a), 100)
    Phi, R_ = np.meshgrid(phi_mesh, r_mesh)
    X_mesh = float(delta) * R_ * np.cos(Phi)
    Z_mesh = R_ * np.sin(Phi)
    # Compute field magnitude on mesh
    Bphi_mesh = Bphi_func(R_, Phi)
    By_mesh = By_func(R_, Phi)
    factor_mesh = R_ * np.sqrt(float(delta)**2 * np.sin(Phi)**2 + np.cos(Phi)**2)
    Bmag_mesh = np.sqrt((Bphi_mesh * factor_mesh)**2 + By_mesh**2)
    cf = ax1.contourf(X_mesh, Z_mesh, Bmag_mesh, cmap="plasma", levels=50)
    fig.colorbar(cf, ax=ax1)
    # Overlay synthetic trajectories
    for z0, (x_vals, _) in zip(z0_vals, trajectories):
        # Dashed for z0 > 0, solid for z0 <= 0
        linestyle = '--' if z0 > 0 else '-'
        ax1.plot(x_vals, np.full_like(x_vals, z0), lw=1.5, linestyle=linestyle)
    ax1.set_title("Synthetic trajectories on |B| contour")
    ax1.set_aspect('equal')
    # Remove distance scale ticks on contour overlay
    ax1.set_xticks([])
    ax1.set_yticks([])

    # --- Bottom: |B| along each trajectory ---
    for z0, (x_vals, Bmag_vals) in zip(z0_vals, trajectories):
        # Dashed for z0 > 0, solid for z0 <= 0
        linestyle = '--' if z0 > 0 else '-'
        ax2.plot(x_vals, Bmag_vals, label=f"z = {z0:.2e}", linestyle=linestyle)
    ax2.set_xlabel("X position")
    # Remove distance scale ticks on the X-axis of magnitude plot
    ax2.set_xticks([])
    ax2.set_ylabel("|B| (T)")
    ax2.set_title("Field magnitude along trajectories")
    ax2.legend(fontsize="small", loc="best")

    plt.tight_layout()
    plt.show()
    fig.savefig("fig_synthetic.svg", format="svg")


# --- Generar reporte Markdown ---
with open("reporte_completo.md", "w") as f:
    f.write("# Reporte de Resultados\n\n")
    f.write("## Magnitudes Físicas y Valores Geométricos\n\n")
    for _, row in df_report.iterrows():
        f.write(f"- **{row['Parametro']}**: {row['Valor']}\n")
    f.write(f"- **R** (semieje menor): {R}\n")
    f.write(f"- **a** (semieje mayor): {a}\n")
    f.write(f"- **L** (longitud del cilindro): {L}\n")
    f.write(f"- **Δ** (delta): {delta}\n\n")
    f.write("## Expresiones Analíticas\n\n")
    for nombre, expr in expresiones.items():
        # Write each expression with its name and LaTeX value
        f.write(f"- **{nombre}** = $$ {sp.latex(expr)} $$\n\n")
    f.write("## Figuras\n\n")
    f.write("![Campos magnéticos](fig1.svg)\n\n")
    f.write("![Corrientes](fig2.svg)\n\n")
    f.write("![Fuerzas de Lorentz](fig3.svg)\n\n")
    f.write("![Corriente Perp/Energía](fig4.svg)\n\n")
    f.write("![Curvas Paramétricas](fig_curve.svg)\n\n")
    f.write("![Ángulo de Pitch](fig_pitch.svg)\n\n")
    f.write("![Trayectorias Sintéticas](fig_synthetic.svg)\n")
print("✅ Reporte Markdown guardado en 'reporte_completo.md'")