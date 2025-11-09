"""
Aerodynamics & Fluid Mechanics Coursework 2025 — Demir Kucukdemiral
This script visualises streamlines around:
  (i) a rotating/lifting cylinder
  (ii) a Joukowski airfoil via the Joukowski transform

Functions:
- compGamma(U_inf, a, alpha, beta=0)
- flow_around_cylinder(U_inf, Gamma, a=1.0, grid_size=400)
- joukowski_airfoil_streamlines(U=1.0, alpha=0.0, beta=0.0, plot=True)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path

def compGamma(U_inf, a, alpha, beta=0.0): 
    """
    Circulation that enforces Kutta condition for a shifted circle (maps to lifting airfoil).
    U_inf: free-stream speed
    a:     circle radius
    alpha: angle of attack (rad)
    beta:  circle-offset direction (rad)
    """
    return 4.0 * np.pi * a * U_inf * np.sin(alpha + beta)

def flow_around_cylinder(U_inf, Gamma, a=1.0, grid_size=400):
    """
    Streamlines around a circular cylinder with circulation Gamma.
    """
    x = np.linspace(-3.0, 3.0, grid_size)
    y = np.linspace(-3.0, 3.0, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    # Complex potential and streamfunction
    F = U_inf * (Z + a**2 / Z) - 1j * Gamma / (2*np.pi) * np.log(Z)
    psi = np.imag(F)

    # Mask inside the cylinder
    psi[np.abs(Z) < a] = np.nan

    plt.figure(figsize=(7, 7))
    plt.contour(X, Y, psi, levels=30)
    th = np.linspace(0, 2*np.pi, 400)
    plt.plot(a*np.cos(th), a*np.sin(th), 'k', linewidth=2)
    plt.xlabel('$x$'); plt.ylabel('$y$')
    plt.title('Streamlines around a circular cylinder with circulation')
    plt.axis('equal'); plt.grid(True); plt.tight_layout()
    plt.show()

def joukowski_airfoil_streamlines(U=1.0, alpha=0.0, beta=0.0, plot=True):
    """
    Streamlines around a Joukowski airfoil created by mapping a shifted circle:
       ζ = z + c^2 / z
    The circle centre is z_c, radius a, chosen so the circle passes through z = c.
    The required circulation (Kutta condition) is Γ = 4π U a sin(α+β).

    Returns:
      Lift per unit span L' = ⍺ U Γ with ρ = 1.225 kg/m^3.
    """
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    r_max = 3.0
    nr, nth = 320, 1800
    levels = 100
    branch = 'outside'

    c = 1.0
    offset_scale = 1.1
    zc = (c - offset_scale*np.cos(beta)) + 1j*(offset_scale*np.sin(beta))

    a = abs(c - zc)

    Gamma = compGamma(U, a, alpha, beta)

    c_eff = max(1e-6, abs(a - abs(zc))) * 0.99

    r = np.linspace(a*1.001, r_max, nr)
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    R, TH = np.meshgrid(r, th)

    z_rel = R * np.exp(1j*TH)
    z_abs = z_rel + zc

    psi = U * (R - a**2 / R) * np.sin(TH - alpha) - (Gamma / (2*np.pi)) * np.log(R)

    # Keep points on the selected branch (avoid the small circle around the origin in mapped plane)
    if branch == 'outside':
        keep = (np.abs(z_abs) >= c_eff) 
    else:
        (np.abs(z_abs) <= c_eff)
    z_abs = z_abs[keep]
    psi   = psi[keep]

    # Joukowski map to airfoil plane
    zeta = z_abs + (c**2)/z_abs
    x, y = np.real(zeta), np.imag(zeta)

    # Airfoil boundary (mapped image of the generating circle)
    theta = np.linspace(0, 2*np.pi, 1600, endpoint=False)
    circle = zc + a*np.exp(1j*theta)
    zeta_b = circle + (c**2)/circle
    xb, yb = np.real(zeta_b), np.imag(zeta_b)

    # Mask triangles inside the airfoil for plotting
    tri = mtri.Triangulation(x, y)
    foil_path = Path(np.column_stack((xb, yb)))
    ctr = np.column_stack((x[tri.triangles].mean(axis=1), y[tri.triangles].mean(axis=1)))
    tri.set_mask(foil_path.contains_points(ctr))

    # Kutta–Joukowski lift per unit span
    rho = 1.225
    Lift = rho * U * Gamma

    if plot:
        plt.figure(figsize=(8, 5))
        plt.tricontour(tri, psi, levels=levels, linewidths=1.2)
        plt.plot(xb, yb, 'k', lw=2)
        pad = 0.18*(xb.max() - xb.min())
        plt.xlim(xb.min()-pad, xb.max()+pad)
        plt.ylim(yb.min()-pad, yb.max()+pad)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel(r'$\Re(\zeta)$'); plt.ylabel(r'$\Im(\zeta)$')
        plt.title(f'Joukowski airfoil streamlines (Γ={Gamma:.3f}, U={U:.3f})')
        plt.grid(True); plt.tight_layout(); plt.show()

    return Lift


_ = joukowski_airfoil_streamlines(U=100.0, alpha=-4.0, beta=5.0, plot=True)
# flow_around_cylinder(U_inf=1.0, Gamma=1.0, a=1.0, grid_size=400)


# alpha_values_deg = np.linspace(-4, 4, 10)
# Lifts = []
# for a_deg in alpha_values_deg:
#     Lifts.append(joukowski_airfoil_streamlines(U=1.0, alpha=np.radians(a_deg), beta=0.0, plot=False))
# plt.figure(figsize=(8,5))
# plt.plot(alpha_values_deg, Lifts, marker='o')
# plt.xlabel('Angle of Attack (degrees)')
# plt.ylabel('Lift per unit span (N/m)')
# plt.title('Lift vs Angle of Attack (Joukowski Airfoil)')
# plt.grid(True); plt.tight_layout(); plt.show()