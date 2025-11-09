import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import math
import osqp


class MPC:
    def __init__(self, solver: str):
        self.g = 9.81
        self.m = 1.0
        self.ell = 0.3
        self.J = 0.2 * self.m * self.ell**2

        # constraints
        self.umin = [0, 0]
        self.umax = [0.6*self.m*self.g, 0.6*self.m*self.g]
        self.ymin = [0]  

        # MPC parameters
        self.h = 0.1
        self.Nh = 30   
        self.nx = 6    
        self.nu = 2    

        # Optimization variables
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, self.Nh+1)  # state trajectory
        self.U = self.opti.variable(self.nu, self.Nh)  # control trajectory
        self.e0 = self.opti.parameter(self.nx)  # initial state

        #refs 
        self.xref = np.array([0, 1, 0, 0, 0, 0])  
        self.u_hover = np.array([0.5*self.m*self.g, 0.5*self.m*self.g])  

        #cost function weifgt matrices
        self.Q = np.diag([10, 10, 10, 1, 1, 1])  
        self.R = np.diag([0.1, 0.1])  

        # sim paramas
        self.Tfinal = 10.0
        self.Nt = int(self.Tfinal / self.h)  
        self.t_hist = np.linspace(0, self.Tfinal, self.Nt)  
        self.xhist = np.zeros((self.nx, self.Nt))  
        self.uhist = np.zeros((self.nu, self.Nt-1))  

        # initial states
        self.xhist[:, 0] = np.array([10, 2, math.pi/2, 0, 0, 0])

        self.solver = solver

    def dynamics(self, x, u):
        """ Quadrotor dynamics """
        x_dot = x[3]
        y_dot = x[4]
        theta_dot = x[5]
        force = u[0] + u[1]
        torque = (self.ell / 2) * (u[1] - u[0])

        x_ddot = -(1/self.m) * force * ca.sin(x[2])
        y_ddot = (1/self.m) * force * ca.cos(x[2]) - self.g
        theta_ddot = (1/self.J) * torque

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

            self.opti.subject_to(ca.vertcat(*self.umin) <= self.U[:, k])
            self.opti.subject_to(ca.vertcat(*self.ymin) <= self.X[1, k])  # y position constraint
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
        """ 2D Animation of the quadrotor flight """

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(min(self.xhist[0, :])-1, max(self.xhist[0, :])+1)
        ax.set_ylim(min(self.xhist[1, :])-1, max(self.xhist[1, :])+1)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.grid()

        # bicopter
        w, h_quad = 0.4, 0.1 
        quad_shape = np.array([
            [-w/2, -h_quad/2], [w/2, -h_quad/2], [w/2, h_quad/2], [-w/2, h_quad/2]
        ])

        quad_patch = patches.Polygon(quad_shape, closed=True, facecolor='blue', alpha=0.8)
        ax.add_patch(quad_patch)
        traj_line, = ax.plot([], [], 'r--', linewidth=2)

        def update(frame):
            pos = self.xhist[:2, frame] 
            theta = self.xhist[2, frame]  
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            v_trans = np.dot(quad_shape, R.T) + pos 
            quad_patch.set_xy(v_trans)
            traj_line.set_data(self.xhist[0, :frame], self.xhist[1, :frame])
            return quad_patch, traj_line

        ani = animation.FuncAnimation(fig, update, frames=self.Nt, interval=50)
        plt.show()


if __name__ == '__main__':
    mpc = MPC('ipopt')
    mpc.MPC()
    mpc.plotter()
    mpc.animate()