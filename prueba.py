

"""
Simulation of Debye screening: central point charge in a plasma.
Calculates Debye length and plots the screened potential.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, k, e
from matplotlib.patches import Circle

def compute_debye_length(n0, T):
    """
    Compute Debye length λ_D = sqrt(ε0 * kT / (n0 * e^2))
    :param n0: electron number density (m^-3)
    :param T: electron temperature (K)
    """
    return np.sqrt(epsilon_0 * k * T / (n0 * e**2))

def screened_potential(r, q, lambda_D):
    """
    Screened Coulomb potential: φ(r) = q / (4πε0 r) * exp(-r/λ_D)
    :param r: radial distance array (m)
    :param q: central charge (C)
    :param lambda_D: Debye length (m)
    """
    phi = q / (4 * np.pi * epsilon_0 * r) * np.exp(-r / lambda_D)
    return phi

def main():
    # Plasma parameters
    n0 = 1e18      # electron density in m^-3
    T = 1e4        # electron temperature in K
    q = 1.0 * e    # central charge in C (e times charge unit)

    # Compute Debye length
    lambda_D = compute_debye_length(n0, T)
    print(f"Debye length: {lambda_D:.3e} m")

    # Radial grid (avoid r=0)
    r = np.linspace(lambda_D * 1e-3, lambda_D * 5, 1000)

    # Compute potential
    phi = screened_potential(r, q, lambda_D)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(r / lambda_D, phi / (q / (4 * np.pi * epsilon_0 * lambda_D)), label='Screened potential')
    plt.axvline(1.0, color='k', linestyle='--', label=r'$r = \lambda_D$')
    plt.xlabel(r'$r / \lambda_D$')
    plt.ylabel(r'$\phi(r) / \left(\frac{q}{4\pi\varepsilon_0\lambda_D}\right)$')
    plt.title('Debye Screening of a Point Charge')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot plasma particles in a box with central charge ---
    # Box size in units of Debye length
    Lbox = lambda_D * 5
    N_particles = 2000
    # Random particle positions in box
    x = np.random.uniform(-Lbox/2, Lbox/2, N_particles)
    y = np.random.uniform(-Lbox/2, Lbox/2, N_particles)
    # Plot scatter of particles
    plt.figure(figsize=(6,6))
    plt.scatter(x / lambda_D, y / lambda_D, s=5, alpha=0.7, label='Plasma particles')
    # Plot central charge
    plt.scatter(0, 0, c='red', s=100, marker='*', label='Central charge')
    ax = plt.gca()
    shield = Circle((0, 0), radius=1.0, fill=False, lw=2, linestyle='--', label=r'$r = \lambda_D$')
    ax.add_patch(shield)
    plt.xlim(-Lbox/lambda_D/2, Lbox/lambda_D/2)
    plt.ylim(-Lbox/lambda_D/2, Lbox/lambda_D/2)
    plt.xlabel(r'$x / \lambda_D$')
    plt.ylabel(r'$y / \lambda_D$')
    plt.title('Plasma particles and central charge')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()