import sympy as sp
import numpy as np
from scipy.integrate import quad, dblquad, cumulative_trapezoid
import matplotlib.pyplot as plt

#######################################
### Definir variables simbólicas ######
#######################################
r, phi, u = sp.symbols('r phi u', real=True)

#######################################
### Constantes físicas del sistema ####
#######################################
mu_0 = 4 * sp.pi * 1e-7  
By0 = 1e-16  # Campo magnético axial en el origen
R = 2.0    # Semieje menor
a = 2.0     # Semieje mayor
L = 10      # Longitud del cilindro

#######################################
### Parámetros de ajuste para j #########
#######################################
beta = 1e-9
alfa = 1e-9
delta = R/a

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
j_phi_phi = 0.12000389484301359 * sp.cos(phi)
j_phi_r = -alfa*r
j_y_r = beta*r**2

#######################################
### Cálculo de jr ######################
#######################################
dj_phi_dphi = sp.diff(j_phi_phi, phi)
integral_j_phi_r = sp.integrate(r * j_phi_r, (r, 0, r))
jr = - (dj_phi_dphi / r) * integral_j_phi_r

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

phi_vals = np.linspace(0, 2 * np.pi, 200)
r_vals = np.linspace(0, a, 200)
Phi, R_ = np.meshgrid(phi_vals, r_vals)
X = delta * R_ * np.cos(Phi)
Z = R_ * np.sin(Phi)

Bphi_vals = Bphi_func(R_, Phi)
By_vals = By_func(R_, Phi)
jr_vals = jr_func(R_, Phi)
jphi_vals = jphi_func(R_, Phi)
jy_vals = jy_func(R_, Phi)

def plot_contour(ax, X, Z, values, title):
    c = ax.contourf(X, Z, values, cmap="coolwarm", levels=100)
    fig_tmp = ax.get_figure()
    fig_tmp.colorbar(c, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("$x = \delta r \cos\phi$")
    ax.set_ylabel("$z = r \sin\phi$")
    ax.set_aspect('equal')

# --- FIGURA 1: Campos Magnéticos ---
fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))
plot_contour(axes1[0], X, Z, Bphi_vals*R_, "$B^\\phi$ (Magnetic Field)")
plot_contour(axes1[1], X, Z, By_vals, "$B^y$ (Magnetic Field)")
fig1.savefig("campos_magneticos.svg", format="svg")
plt.show()

# --- FIGURA 2: Corrientes ---------------
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
plot_contour(axes2[0], X, Z, jr_vals, "$j^r$ (Current Field)")
plot_contour(axes2[1], X, Z, jphi_vals*R_, "$j^y$ (Current Field)")
plot_contour(axes2[2], X, Z, jy_vals, "$j^\\phi$ (Current Field)")
fig2.savefig("corrientes.svg", format="svg")
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
f_r_vals = f_r_func(R_, Phi)
f_y_vals = f_y_func(R_, Phi)
f_phi_vals = f_phi_func(R_, Phi)

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
plot_contour(axes3[0], X, Z, f_r_vals, "$f^r$ (Lorentz Force)")
plot_contour(axes3[1], X, Z, f_y_vals, "$f^y$ (Lorentz Force)")
plot_contour(axes3[2], X, Z, f_phi_vals*R_, "$f^\\phi$ (Lorentz Force)")
fig3.savefig("fuerzas_lorentz.svg", format="svg")
plt.show()

# --- FIGURA 4: Corriente Perpendicular y Desalineamiento/Energía -----------
jperp = (g_yy * g_phiphi / (delta * r * sp.sqrt(g_yy*By*By + g_phiphi*Bphi*Bphi))) * (jy * Bphi - jphi * By)
jperp = sp.simplify(jperp)
j_perp_func = sp.lambdify((r, phi), jperp, "numpy")
j_perp_values = j_perp_func(R_, Phi)

fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
plot_contour(axes4[0], X, Z, j_perp_values, "$j_{\\perp}$ (Perpendicular Current)")

modj = sp.sqrt(g_rr*jr*jr + 2*g_rphi*jr*jphi + g_yy*jy*jy + g_phiphi*jphi*jphi)
modB = sp.sqrt(g_yy*By*By + g_phiphi*Bphi*Bphi)
modLorentz = sp.sqrt(g_rr*fr*fr + 2*g_rphi*fr*fphi + g_yy*fy*fy + g_phiphi*fphi*fphi)
miss = modLorentz / (modj * modB)
miss = sp.lambdify((r, phi), miss, "numpy")
miss_values = miss(R_, Phi)
plot_contour(axes4[1], X, Z, miss_values, "$sin(\\omega)$")

magE = (g_yy*By*By + g_phiphi*Bphi*Bphi) / (2*mu_0)
magE = sp.lambdify((r, phi), magE, "numpy")
magE_vals = magE(R_, Phi)
plot_contour(axes4[2], X, Z, magE_vals, "Magnetic Energy Density")
fig4.savefig("corriente_desalineamiento_energia.svg", format="svg")
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
def compute_line_xyz(r_fixed, By_func, Bphi_func, delta, n_points=300):
    phi_array = np.linspace(0, 2*np.pi, n_points)
    integrand = r_fixed * abs(By_func(r_fixed, phi_array) / Bphi_func(r_fixed, phi_array))
    z_array = cumulative_trapezoid(integrand, phi_array, initial=0.0)
    x_array = delta * r_fixed * np.cos(phi_array)
    y_array = r_fixed * np.sin(phi_array)
    return x_array, y_array, z_array

def draw_elliptical_cylinder(ax, R, L, delta, n_phi=30, n_z=30, alpha=0.15):
    phi_cyl = np.linspace(0, 2*np.pi, n_phi)
    z_cyl   = np.linspace(0, L, n_z)
    Phi_cyl, Z_cyl = np.meshgrid(phi_cyl, z_cyl)
    X_cyl = delta * R * np.cos(Phi_cyl)
    Y_cyl = R * np.sin(Phi_cyl)
    ax.plot_wireframe(X_cyl, Y_cyl, Z_cyl, color='gray', alpha=alpha)

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
for r_fixed in r_values:
    x_arr, y_arr, z_arr = compute_line_xyz(r_fixed, By_func, Bphi_func, delta)
    ax3d_front.plot(x_arr, y_arr, z_arr, label=f"r={r_fixed}")
    ax3d_side.plot(x_arr, y_arr, z_arr)
    ax_xy.plot(x_arr, y_arr)
    ax_xz.plot(x_arr, z_arr)
    ax_yz.plot(y_arr, z_arr)

draw_elliptical_cylinder(ax3d_front, R, L, delta, alpha=0.2)
draw_elliptical_cylinder(ax3d_side, R, L, delta, alpha=0.2)
ax3d_front.view_init(elev=20, azim=-60)
ax3d_side.view_init(elev=20, azim=30)
set_axes_equal_3d(ax3d_front)
set_axes_equal_3d(ax3d_side)
ax_xy.set_aspect('equal', adjustable='datalim')
ax_xz.set_aspect('equal', adjustable='datalim')
ax_yz.set_aspect('equal', adjustable='datalim')
ax3d_front.legend()
fig5.savefig("lineas_de_campo.svg", format="svg")
plt.tight_layout()
plt.show()
