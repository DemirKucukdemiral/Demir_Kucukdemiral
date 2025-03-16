import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math
import osqp


class MPC:
    def __init__(self, solver: str):
        self.G = 6.67430e-11
        self.M = 5.972e24
        self.R = 8371000  
        self.M = 5.972e24
        self.g = 9.81
        self.m = 30.0
        self.ell = 13.0
        self.radius = 3.0
        self.J = 0.25 * self.m * self.radius**2 + 1/12 * self.m * self.ell**2

        self.F_g = self.G * self.M *self.m / self.R**2

        # constraints
        self.umin = [0, -np.pi, -30]
        self.umax = [900, np.pi, 30]


        # MPC parameters
        self.h = 0.05 
        self.Nh = 50   
        self.nx = 6    
        self.nu = 3   

        # Optimization variables
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.Nh+1)  # state trajectory
        self.U = self.opti.variable(self.nu, self.Nh)  # control trajectory
        self.e0 = self.opti.parameter(self.nx)  # initial state

        #refs 
        self.xref = np.array([100, 100, -np.pi/2, 10, 0, 0])  
        self.u_hover = np.array([0, 0, 0])  

        #cost function weifgt matrices
        self.Q = np.diag([30, 30, 30, 10, 10, 10])  
        self.R = np.diag([0.1, 0.1, 0.1])  
        self.Qf = np.diag([1000, 1000, 1000, 100, 100, 100])

        # sim paramas
        self.Tfinal = 20.0
        self.Nt = int(self.Tfinal / self.h)  
        self.t_hist = np.linspace(0, self.Tfinal, self.Nt)  
        self.xhist = np.zeros((self.nx, self.Nt))  
        self.uhist = np.zeros((self.nu, self.Nt-1))  

        # initial states
        self.xhist[:, 0] = np.array([0, 0, -math.pi/2, 0, 0, 0])
        

        self.solver = solver

    def dynamics(self, x, u):
        """ Quadrotor dynamics """
        x_dot = x[3]
        y_dot = x[4]
        theta_dot = x[5]

        torque = u[0]*ca.sin(u[1]) + u[2]

        x_ddot = u[0]/self.m * ca.cos(x[3]+u[1])
        y_ddot = u[0]/self.m * ca.sin(x[3]+u[1]) 
        theta_ddot = (self.ell/(2*self.J)) * torque

        return ca.vertcat(x_dot, y_dot, theta_dot, x_ddot, y_ddot, theta_ddot)

    def rk4_step(self, x, u, dt):
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + dt/2 * k1, u)
        k3 = self.dynamics(x + dt/2 * k2, u)
        k4 = self.dynamics(x + dt * k3, u)
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def MPC(self):
        self.opti.subject_to(self.X[:, 0] == self.e0)
        cost = 0  

        for k in range(self.Nh):
            x_next = self.rk4_step(self.X[:, k], self.U[:, k], self.h)
            self.opti.subject_to(self.X[:, k+1] == x_next)

            cost += ca.mtimes([(self.X[:, k] - self.xref).T, self.Q, (self.X[:, k] - self.xref)])[0, 0]
            cost += ca.mtimes([self.U[:, k].T, self.R, self.U[:, k]])[0, 0]
            cost += ca.mtimes([(self.X[:, self.Nh] - self.xref).T, self.Qf, (self.X[:, self.Nh] - self.xref)])[0, 0]

            self.opti.subject_to(ca.vertcat(*self.umin) <= self.U[:, k])
            self.opti.subject_to(self.U[:, k] <= ca.vertcat(*self.umax))

        self.opti.minimize(cost)
        self.opti.solver(self.solver)

        def mpc_control(x0):
            self.opti.set_value(self.e0, x0)
            sol = self.opti.solve()
            return np.array(sol.value(self.U[:, 0]))  

        for k in range(self.Nt-1):
            u = mpc_control(self.xhist[:, k])
            self.uhist[:, k] = u  
            self.xhist[:, k+1] = np.array(self.rk4_step(self.xhist[:, k], u, self.h)).flatten()

    def plotter(self):
        """ Plot x, y, and theta over time """
        plt.figure()
        plt.plot(self.t_hist, self.xhist[0, :], label='x')
        plt.plot(self.t_hist, self.xhist[1, :], label='y')
        plt.plot(self.t_hist, self.xhist[2, :], label='theta')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('State')
        plt.title('State Trajectory')
        plt.grid()
        plt.show()

    def animate(self):
        """ Animate the rocket trajectory with a box+triangle for the rocket and a star for the target """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        # Set axis limits to see the region of interest
        ax.set_xlim(-20, 120)
        ax.set_ylim(-20, 120)
        
        # Draw the target as a star (using a marker)
        target = self.xref  # target position (x, y)
        target_plot, = ax.plot(target[0], target[1], marker="*", markersize=15, color="gold", label="Target")

        # Create patches for the rocket body (a rectangle) and nose (a triangle)
        # Define rocket dimensions (these values can be adjusted)
        body_width = 5
        body_height = 15
        # The rectangle is centered at (0,0); we'll transform it later.
        rocket_body = patches.Rectangle((-body_width/2, -body_height/2), body_width, body_height,
                                        fc="skyblue", ec="blue")
        ax.add_patch(rocket_body)

        # Define a triangle for the nose (pointing upward in its local frame)
        nose_height = 5
        # Triangle vertices (local coordinates): bottom left, bottom right, tip.
        nose_coords = np.array([[-body_width/2, body_height/2],
                                [ body_width/2, body_height/2],
                                [0, body_height/2 + nose_height]])
        rocket_nose = patches.Polygon(nose_coords, closed=True, fc="red", ec="darkred")
        ax.add_patch(rocket_nose)

        # Function to update the rocket drawing
        def update(frame):
            # Get current state from history
            state = self.xhist[:, frame]
            x_pos, y_pos, theta = state[0], state[1], state[2]
            
            # Create a transformation: rotate by theta (in degrees) and translate to (x_pos, y_pos)
            t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(0, 0, np.degrees(theta))
            t = t.translate(x_pos, y_pos)
            
            # Update rocket body and nose transformations
            rocket_body.set_transform(t + ax.transData)
            rocket_nose.set_transform(t + ax.transData)
            
            return rocket_body, rocket_nose, target_plot

        ani = animation.FuncAnimation(fig, update, frames=self.Nt, interval=50, blit=True, repeat=True, repeat_delay=2000)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Rocket Trajectory Animation")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    mpc = MPC('ipopt')
    mpc.MPC()
    mpc.plotter()
    mpc.animate()