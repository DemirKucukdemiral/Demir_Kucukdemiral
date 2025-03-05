import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Physical / System Parameters
# ---------------------------------------------------------
M = 1.0    # cart mass [kg]
m = 0.1    # pendulum mass [kg]
l = 0.5    # pendulum pivot-to-CoM length [m]
g = 9.81   # gravity [m/s^2]
dt = 0.02  # sampling time [s]

# Original (4x4) continuous-time linearization around theta=0
A_c = np.array([
    [0,         1,                0,                 0],
    [0,         0,       (m*g)/M,                 0],
    [0,         0,                0,                 1],
    [0,         0, (M+m)*g/(M*l),                 0]
])
B_c = np.array([
    [0],
    [1/M],
    [0],
    [1/(M*l)]
])

# Discretize (Euler)
A_d_4 = np.eye(4) + dt * A_c
B_d_4 = dt * B_c

# ---------------------------------------------------------
# 2. Build the Augmented System with Integrator
# ---------------------------------------------------------
# Augment dimension: x_aug = [x(4x1), xI] in R^5
# xI(k+1) = xI(k) + dt*x_cart(k)

# We'll form A_hat and B_hat for the augmented system:
A_hat = np.block([
    [A_d_4,            np.zeros((4,1))],
    [dt*np.array([1,0,0,0]),         1]
])
B_hat = np.vstack([B_d_4, np.array([[0]])])

# ---------------------------------------------------------
# 3. MPC Setup
# ---------------------------------------------------------
N = 20  # Prediction horizon

# Cost on state: We'll make a 5x5 Q_hat that includes:
# - The original 4x4 Q for x
# - A penalty for the integrator state xI
# Example: penalize xI so we don't let integrator drift
Q_4 = np.diag([1.0, 0.0, 10.0, 0.0])  # for [x, xdot, theta, thetadot]
Q_i = np.array([[5.0]])             # penalty for integrator state
Q_hat = np.block([
    [Q_4,              np.zeros((4,1))],
    [np.zeros((1,4)),  Q_i         ]
])

# Same input cost
R = np.array([[0.1]])

# Define a reference for the cart position (and angle).
# Here we assume we want to track cart position=0, angle=0.
# If we wanted x_ref=1, we could incorporate that by changing the integrator equation,
# or offsetting states. We'll keep it simple at x_ref=0, theta_ref=0.
def mpc_control(X_current):
    """
    Given the current augmented state X_current in R^5,
    solve the MPC problem and return the first control input.
    """
    # Decision variables: X in R^(5 x (N+1)), u in R^(1 x N)
    X_var = cp.Variable((5, N+1))
    U_var = cp.Variable((1, N))

    cost = 0
    constraints = []

    # Initial condition
    constraints.append(X_var[:, 0] == X_current)

    # Build cost/constraints
    for k in range(N):
        # Quadratic cost on states and inputs
        cost += cp.quad_form(X_var[:, k], Q_hat) + cp.quad_form(U_var[:, k], R)

        # System dynamics: X_{k+1} = A_hat X_k + B_hat U_k
        constraints.append(X_var[:, k+1] == A_hat @ X_var[:, k] + B_hat @ U_var[:, k])

        # Input constraints (example)
        constraints.append(U_var[:, k] >= -10.0)
        constraints.append(U_var[:, k] <=  10.0)

        # Constrain the cart position (x_cart = X_var[0, k]) to be between [0, 2]
        constraints.append(X_var[0, k] >= 0.0)
        constraints.append(X_var[0, k] <= 2.0)

    # Terminal cost
    cost += cp.quad_form(X_var[:, N], Q_hat)

    # Also constrain the final cart position:
    constraints.append(X_var[0, N] >= 0.0)
    constraints.append(X_var[0, N] <= 2.0)

    # Solve the QP
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
    except cp.SolverError:
        # fallback
        prob.solve(solver=cp.ECOS, verbose=False)

    # Check status
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"WARNING: Solver status {prob.status}. Returning zero control.")
        return 0.0

    return U_var.value[0, 0]

# ---------------------------------------------------------
# 4. Simulation
# ---------------------------------------------------------
sim_time = 5.0
num_steps = int(sim_time / dt)

# Initial state (4-dim) for the real system
x_real = np.array([0.5, 0.0, 5.0*np.pi/180.0, 0.0])  # cart at x=0.5, small angle
xI_init = 0.0  # integrator initial

# Augmented state [ x_real; xI ]
X_aug = np.concatenate([x_real, [xI_init]])
history_state = [x_real.copy()]
history_intg = [xI_init]
control_history = []

for t in range(num_steps):
    # Solve MPC for the augmented state
    u = mpc_control(X_aug)

    # Apply to the real (linear discrete) system for the real states
    x_real = A_d_4 @ x_real + B_d_4.flatten() * u
    
    # Update integrator manually to keep X_aug consistent
    xI_new = X_aug[4] + dt * x_real[0]  # integrator of cart position
    
    # Update the augmented state
    X_aug = np.concatenate([x_real, [xI_new]])

    # Store data
    history_state.append(x_real.copy())
    history_intg.append(xI_new)
    control_history.append(u)

history_state = np.array(history_state)
history_intg = np.array(history_intg)
time_axis = np.arange(num_steps+1)*dt

# ---------------------------------------------------------
# 5. Plot Results
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Cart position
plt.subplot(4,1,1)
plt.plot(time_axis, history_state[:,0], label='Cart Position (m)')
plt.axhline(0, color='k', linewidth=0.8)
plt.axhline(2, color='k', linewidth=0.8, linestyle='--')
plt.title('MPC with Integral Action')
plt.ylabel('x (m)')
plt.grid(True)
plt.legend()

# Pendulum angle
plt.subplot(4,1,2)
plt.plot(time_axis, np.rad2deg(history_state[:,2]), label='Pendulum Angle (deg)')
plt.axhline(0, color='k', linewidth=0.8)
plt.ylabel('theta (deg)')
plt.grid(True)
plt.legend()

# Integrator state
plt.subplot(4,1,3)
plt.plot(time_axis, history_intg, label='Integrator State')
plt.axhline(0, color='k', linewidth=0.8)
plt.ylabel('Integrator Value')
plt.grid(True)
plt.legend()

# Control input
plt.subplot(4,1,4)
plt.step(time_axis[:-1], control_history, where='post', label='Control Input (N)')
plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('u (N)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()