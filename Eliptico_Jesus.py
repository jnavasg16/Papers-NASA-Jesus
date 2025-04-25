import sympy as sp
import numpy as np
from scipy.integrate import quad, dblquad, cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Nuevo import para colormaps “inferno”
import matplotlib.ticker as ticker

import pandas as pd

# --- Configuración de Modo ---
modo_rapido = True

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
R = 0.7e10    # Semieje menor
a = 1e10    # Semieje mayor
L = 10     # Longitud del cilindro
                                                                 #OK
#######################################
### Parámetros de ajuste para j #########
#######################################

alfa = 1e-32
beta = -alfa*a*10
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
j_phi_phi = 0.12615662610100803 \
    + (-0.24000778968602718) * sp.cos(1 * phi) \
    + (0.20657661898691135) * sp.cos(2 * phi) \
    + (-0.16088203263124973) * sp.cos(3 * phi) \
    + (0.11337165224497914) * sp.cos(4 * phi) \
    + (-0.07228895706727245) * sp.cos(5 * phi) \
    + (0.041707100072565964) * sp.cos(6 * phi) \
    + (-0.0217730154538321) * sp.cos(7 * phi) \
    + (0.010284844252703521) * sp.cos(8 * phi) \
    + (-0.004395896006372525) * sp.cos(9 * phi) \
    + (0.0017000733205040617) * sp.cos(10 * phi)

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

flujo_poloidal = L * sp.integrate(Bphi_old * delta * r, (r, 0, R))

#######################################
### Corriente axial poloidal ###########
#######################################

Bphi_old_cc = - (mu_0 * delta / r**2) * sp.integrate(r * j_y_r, (r, 0, r))
# Convertir la expresión simbólica en una función numérica:
Bphi_old_cc_func = sp.lambdify(r, Bphi_old_cc, "numpy")
integrand_cc = lambda r: Bphi_old_cc_func(r) * r
flujo_cc, error = quad(integrand_cc, 0, R)
j_y_phi = - (flujo_cc / flujo_poloidal)

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
    phi_vals = np.linspace(0, 2 * np.pi, 100)
    r_vals = np.linspace(epsilon, a, 100)
else:
    phi_vals = np.linspace(0, 2 * np.pi, 200)
    r_vals = np.linspace(epsilon, a, 200)
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
    ax.set_xlabel(r"$x = \delta r \cos\phi$", fontsize=12)
    ax.set_ylabel(r"$z = r \sin\phi$", fontsize=12)
    ax.set_aspect('equal')
    
# --- FIGURA 1: Campos Magnéticos ---
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
fig1.tight_layout()
plot_contour(axes1[0], X, Z, Bphi_vals*factor, "$B^\\phi$ (Magnetic Field)")
plot_contour(axes1[1], X, Z, By_vals, "$B^y$ (Magnetic Field)")
plt.show()

# --- FIGURA 2: Corrientes ---------------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.tight_layout()
plot_contour(axes2[0], X, Z, jr_vals, "$j^r$ (Current Field)")
plot_contour(axes2[1], X, Z, jy_vals, "$j^y$ (Current Field)")
plot_contour(axes2[2], X, Z, jphi_vals*factor, "$j^\\phi$ (Current Field)")
plt.show()

# --- FIGURA 3: Fuerzas de Lorentz -------
fr = (g_yy * g_phiphi) / (delta * r) * (jy * Bphi - jphi * By)
fy = (1 / (delta * r)) * ((g_phiphi * jphi + g_rphi * jr) * g_rphi * Bphi - (g_rphi * jphi + g_rr * jr) * g_phiphi * Bphi)
fphi = (g_rr * g_yy) / (delta * r) * jr * By
fr = sp.simplify(fr)
fy = sp.simplify(fy)
fphi = sp.simplify(fphi)
f_r_func = sp.lambdify((r, phi), fr, "numpy")
f_y_func = sp.lambdify((r, phi), fy, "numpy")
f_phi_func = sp.lambdify((r, phi), fphi, "numpy")
f_r_vals_raw = f_r_func(R_, Phi)
if np.ndim(f_r_vals_raw) == 0:
    f_r_vals = np.full_like(R_, f_r_vals_raw)
else:
    f_r_vals = f_r_vals_raw

f_y_vals_raw = f_y_func(R_, Phi)
if np.ndim(f_y_vals_raw) == 0:
    f_y_vals = np.full_like(R_, f_y_vals_raw)
else:
    f_y_vals = f_y_vals_raw

f_phi_vals_raw = f_phi_func(R_, Phi)
if np.ndim(f_phi_vals_raw) == 0:
    f_phi_vals = np.full_like(R_, f_phi_vals_raw)
else:
    f_phi_vals = f_phi_vals_raw

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.tight_layout()
plot_contour(axes3[0], X, Z, f_r_vals, "$f^r$ (Lorentz Force)")
plot_contour(axes3[1], X, Z, f_y_vals, "$f^y$ (Lorentz Force)")
plot_contour(axes3[2], X, Z, f_phi_vals*factor, "$f^\\phi$ (Lorentz Force)")
plt.show()

# --- FIGURA 4: Corriente Perpendicular y Desalineamiento/Energía -----------
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

modj = sp.sqrt(g_rr*jr*jr + 2*g_rphi*jr*jphi + g_yy*jy*jy + g_phiphi*jphi*jphi)
modB = sp.sqrt(g_yy*By*By + g_phiphi*Bphi*Bphi)
modLorentz = sp.sqrt(g_rr*fr*fr + 2*g_rphi*fr*fphi + g_yy*fy*fy + g_phiphi*fphi*fphi)
miss = modLorentz / (modj * modB)
miss = sp.lambdify((r, phi), miss, "numpy")
miss_raw = miss(R_, Phi)
if np.ndim(miss_raw) == 0:
    miss_values = np.full_like(R_, miss_raw)
else:
    miss_values = miss_raw
plot_contour(axes4[1], X, Z, miss_raw, "$sin(\\omega)$")

magE = (g_yy*By*By + g_phiphi*Bphi*Bphi) / (2*mu_0)
magE = sp.lambdify((r, phi), magE, "numpy")
magE_raw = magE(R_, Phi)
if np.ndim(magE_raw) == 0:
    magE_vals = np.full_like(R_, magE_raw)
else:
    magE_vals = magE_raw
plot_contour(axes4[2], X, Z, magE_vals, "Magnetic Energy Density")
plt.show()

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
        result, err = quad(lambda phip: By_func(rp, phip), lower, upper)
        return result
    result_outer, err = quad(lambda rp: rp * inner_integral(rp), 0, r_val)
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

H_numeric, err = dblquad(integrand_for_dblquad,
                         0, R,
                         lambda r_val: 0,
                         lambda r_val: 2*np.pi)
print("Helicidad Magnética: ", H_numeric)



# --- FIGURA 5: Líneas de campo en 3D y proyecciones 2D ---
def compute_line_xyz(r_fixed, By_func, Bphi_func, delta, L, n_points=300):
    # Estimate z-growth over one full turn
    phi_single = np.linspace(0, 2*np.pi, n_points)
    integrand_single = r_fixed * np.abs(By_func(r_fixed, phi_single) / Bphi_func(r_fixed, phi_single))
    z_single = cumulative_trapezoid(integrand_single, phi_single, initial=0.0)
    z_end = z_single[-1] if z_single.size > 0 else 0.0
    # Determine number of turns to reach height L
    num_turns = (L / z_end) if (z_end > 0) else 1.0
    total_points = max(int(n_points * num_turns), n_points)
    phi_array = np.linspace(0, 2 * np.pi * num_turns, total_points)
    integrand = r_fixed * np.abs(By_func(r_fixed, phi_array) / Bphi_func(r_fixed, phi_array)) / 1
    z_array = cumulative_trapezoid(integrand, phi_array, initial=0.0)
    # Cap z at L
    z_array = np.clip(z_array, 0, L)
    x_array = delta * r_fixed * np.cos(phi_array)
    y_array = r_fixed * np.sin(phi_array)
    return x_array, y_array, z_array

def draw_elliptical_cylinder(ax, R, L, delta, n_phi=30, n_z=30, alpha=0.02):
    phi_cyl = np.linspace(0, 2*np.pi, n_phi)
    z_cyl   = np.linspace(0, L, n_z)
    Phi_cyl, Z_cyl = np.meshgrid(phi_cyl, z_cyl)
    X_cyl = delta * R * np.cos(Phi_cyl)
    Y_cyl = R * np.sin(Phi_cyl)
    # Cilindro en gris, muy transparente
    ax.plot_wireframe(
        X_cyl, Y_cyl, Z_cyl,
        color="grey",
        alpha=alpha
    )

def set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_mid   = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid   = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid   = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])

fig5 = plt.figure(figsize=(18, 12))
gs0 = fig5.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.3)
top_gs = gs0[0].subgridspec(nrows=1, ncols=2, wspace=0.3)
bottom_gs = gs0[1].subgridspec(nrows=1, ncols=3, wspace=0.3)

ax3d_front = fig5.add_subplot(top_gs[0], projection='3d')
ax3d_front.set_title("Front view")
ax3d_front.set_xlabel("x")
ax3d_front.set_ylabel("z")
ax3d_front.set_zlabel("y")
ax3d_side  = fig5.add_subplot(top_gs[1], projection='3d')
ax3d_side.set_title("Lateral view")
ax3d_side.set_xlabel("x")
ax3d_side.set_ylabel("z")
ax3d_side.set_zlabel("y")

ax_xy = fig5.add_subplot(bottom_gs[0])
ax_xy.set_title("XZ projection")
ax_xy.set_xlabel("x")
ax_xy.set_ylabel("z")
ax_xz = fig5.add_subplot(bottom_gs[1])
ax_xz.set_title("XY projection")
ax_xz.set_xlabel("x")
ax_xz.set_ylabel("y")
ax_yz = fig5.add_subplot(bottom_gs[2])
ax_yz.set_title("ZY projection")
ax_yz.set_xlabel("z")
ax_yz.set_ylabel("y")

r_values = [R/4, R/2, R]
# Asignar colores fijos cíclicamente para las líneas
line_colors = ["orange", "red", "yellow"]
for idx, r_fixed in enumerate(r_values):
    color = line_colors[idx % len(line_colors)]
    x_arr, y_arr, z_arr = compute_line_xyz(r_fixed, By_func, Bphi_func, delta, L)
    # Export up to 500 points of the curve to CSV
    n_pts = len(x_arr)
    if n_pts > 500:
        idxs = np.linspace(0, n_pts - 1, 500, dtype=int)
        x_out = x_arr[idxs]
        y_out = y_arr[idxs]
        z_out = z_arr[idxs]
    else:
        x_out, y_out, z_out = x_arr, y_arr, z_arr
    df = pd.DataFrame({'x': x_out, 'y': y_out, 'z': z_out})
    df.to_csv(f'curve_r_{r_fixed:.2f}.csv', index=False)
    ax3d_front.plot(x_arr, y_arr, z_arr, label=f"r={r_fixed}", color=color)
    ax3d_side.plot(x_arr, y_arr, z_arr, color=color)
    ax_xy.plot(x_arr, y_arr, color=color)
    ax_xz.plot(x_arr, z_arr, color=color)
    ax_yz.plot(y_arr, z_arr, color=color)

draw_elliptical_cylinder(ax3d_front, R, L, delta, alpha=0.02)
draw_elliptical_cylinder(ax3d_side, R, L, delta, alpha=0.02)
ax3d_front.view_init(elev=20, azim=-60)
ax3d_side.view_init(elev=20, azim=30)
set_axes_equal_3d(ax3d_front)
set_axes_equal_3d(ax3d_side)
ax_xy.set_aspect('equal', adjustable='datalim')
ax_xz.set_aspect('equal', adjustable='datalim')
ax_yz.set_aspect('equal', adjustable='datalim')
ax3d_front.legend()
plt.tight_layout()
plt.show()

# --- Generar Reporte Completo de Magnitudes ---
from scipy.integrate import dblquad


mag_energy_total = dblquad(lambda phi, r: magE(r, phi) * delta * r, 0, R, lambda r: 0, lambda r: 2*np.pi)[0]

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
