"""
======================================================================
Aero_2.py
---------
This script is developed by Demir Kucukdemiral 2883935K for 
Fluid Mechanics & Aerodynamics Coursework 2025.

======================================================================

This script visualises streamlines around:
  (i) a rotating/lifting cylinder
  (ii) a Joukowski airfoil via the Joukowski transform

Functions:
- compGamma(U_inf, a, alpha, beta)
- flow_around_cylinder(U_inf, Gamma, a)
- joukowski_airfoil_streamlines(U=1.0, alpha=0.0, beta=0.0, plot=True)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path

def compGamma(U_inf, a, alpha, beta=0.0): 
    """
    calculates the circulation around an offset circle using the Kutta condition
    U_inf: free-stream speed
    a:     circle radius
    alpha: angle of attack (rad)
    beta:  camber angle (rad)
    """
    return 4.0 * np.pi * a * U_inf * np.sin(alpha + beta)

def flow_around_cylinder(U_inf, Gamma, a=1.0, grid_size=400):
    """
    Streamlines around a circular cylinder with circulation
    U_inf: free-stream vel
    Gamma: circulation around the cylinder
    a:     cylinder radius 
    """
    x = np.linspace(-3.0, 3.0, grid_size)
    y = np.linspace(-3.0, 3.0, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    # complex potential function
    F = U_inf * (Z + a**2 / Z) + 1j * Gamma / (2*np.pi) * np.log(Z)
    psi = np.imag(F) # stream function 

    psi[np.abs(Z) < a] = np.nan # getting rid of inner points of the cylinder

    plt.figure(figsize=(7, 7))

    plt.contour(X, Y, psi,levels=30, colors='b', linestyles='solid',linewidths=1.2)

    th = np.linspace(0, 2*np.pi, 400)
    plt.plot(a * np.cos(th), a * np.sin(th), 'k', linewidth=2)

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def joukowski_airfoil_streamlines(U=1.0, alpha=0.0, beta=0.0, plot=True):
    """
    Streamlines around a Joukowski airfoil created by mapping a shifted circle 
    using the jukowski transform
    U:     free-stream  vel
    alpha: angle of attack (deg)
    beta:  camber angle (deg)
    plot:  can be set to false when computing only the lift

    Returns:
      Lift per unit span (L')
    """
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    r_max = 3.0
    nr, nth = 1000, 2000
    levels = 50
    branch = 'outside'

    c = 1.0 #scale factor
    offset_scale = 1.1

    zc = (c - offset_scale*np.cos(beta)) + 1j*(offset_scale*np.sin(beta))

    print(f'Circle centre zc = {zc:.3f} at angle of attack alpha = {np.degrees(alpha):.2f} deg')

    a = abs(c - zc) 
    print(f'Circle radius a = {a:.3f}')
    Gamma = compGamma(U, a, alpha, beta) 

    c_eff = max(1e-6, abs(a - abs(zc))) * 0.99 # effective radius to avoid singularity 

    r = np.linspace(a*1.001, r_max, nr)
    th = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    R, TH = np.meshgrid(r, th)

    z_rel = R * np.exp(1j*TH)
    z_abs = z_rel + zc

    psi = U * (R - a**2 / R) * np.sin(TH - alpha) + (Gamma / (2*np.pi)) * np.log(R) #stream function eq using polar coords

    # need to ensure that points inside the aerofoil are removed
    if branch == 'outside':
        keep = (np.abs(z_abs) >= c_eff) 
    else:
        (np.abs(z_abs) <= c_eff)
    z_abs = z_abs[keep]
    psi   = psi[keep]

    # mapping coordinates to the Joukowski aerofoil plane 
    zeta = z_abs + (c**2)/z_abs
    x, y = np.real(zeta), np.imag(zeta)

    # circle points mapped to aerofoil surface
    theta = np.linspace(0, 2*np.pi, 1600, endpoint=False)
    circle = zc + a*np.exp(1j*theta)
    zeta_b = circle + (c**2)/circle
    xb, yb = np.real(zeta_b), np.imag(zeta_b)
    
    # plotting, and masking the inside points of the aerofoil
    if plot == True:
        tri = mtri.Triangulation(x, y)
        foil_path = Path(np.column_stack((xb, yb)))
        ctr = np.column_stack((x[tri.triangles].mean(axis=1), y[tri.triangles].mean(axis=1)))
        tri.set_mask(foil_path.contains_points(ctr))

    rho = 1.225
    Lift = rho * U * Gamma

    chord_length = xb.max() - xb.min()            
    Theory_lift = np.pi * rho * U**2 * chord_length * np.sin(alpha + beta)
    
    psi_body = (Gamma/(2*np.pi))*np.log(a)   
    psi = psi - psi_body

    if plot:
        plt.figure(figsize=(8, 5))

        plt.tricontour(tri, psi,  levels=levels, colors='b', linestyles='solid', linewidths=1.2)

        plt.plot(xb, yb, 'k', lw=1.2)

        pad = 0.25 * (xb.max() - xb.min())
        plt.xlim(xb.min() - pad, xb.max() + pad)
        plt.ylim(yb.min() - pad, yb.max() + pad)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel(r'$\Re(\zeta)$')
        plt.ylabel(r'$\Im(\zeta)$')

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return Lift, Theory_lift

# fow around cylinder, static and rotating
flow_around_cylinder(U_inf=1.0, Gamma=5.0, a=1.0, grid_size=400)
flow_around_cylinder(U_inf=1.0, Gamma=0.0, a=1.0, grid_size=400)

# flow around jukowski qerofoils symmetric and cambered
joukowski_airfoil_streamlines(U=1.0, alpha=5.0, beta=0.0, plot=True)
joukowski_airfoil_streamlines(U=1.0, alpha=5.0, beta=10.0, plot=True)


alpha_values_deg = np.linspace(-5, 10, 16)
Lifts = []
Theory_Lifts = []
for a_deg in alpha_values_deg:
    Lifts.append(joukowski_airfoil_streamlines(U=1.0, alpha=a_deg, beta=10.0, plot=False)[0])
    Theory_Lifts.append(joukowski_airfoil_streamlines(U=1.0, alpha=a_deg, beta=10.0, plot=False)[1])
gradient = np.gradient(Lifts, alpha_values_deg)
theory_gradient = np.gradient(Theory_Lifts, alpha_values_deg)
print(">>>>>>================================================<<<<<<")
print("Computed lift slope dL/dalpha:", gradient)
print("Theoretical lift slope dL/dalpha:", theory_gradient)
plt.figure(figsize=(8,5))
plt.plot(alpha_values_deg, Lifts, marker='o')
plt.plot(alpha_values_deg, Theory_Lifts, color = 'r',marker='x', linestyle='--')
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('Lift per unit span (N/m)')
plt.legend(['Computed Lift', 'Theoretical Lift'], loc='best')
plt.grid(True); plt.tight_layout(); plt.show()