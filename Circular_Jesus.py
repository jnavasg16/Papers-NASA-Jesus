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
R = 1e10    # Semieje menor
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
j_phi_phi = 1

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



