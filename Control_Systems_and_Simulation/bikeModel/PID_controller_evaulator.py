import numpy as np


"""
States:
    X[0]: lateral velocity
    x[1]:yaw
    x[2]:yaw rate
Control inputs:
    u = steering angle
"""


class Simulator:
    def __init__(self, Kp, Ki, Kd, disturbance_percentage, dt, Cf, Cr, lf, lr, Iz, m, I_axle, r):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.disturbance = disturbance_percentage
        self.I = 0.0
        self.previous_error = 0.0
        self.dt = dt
        self.Cf = Cf
        self.Cr = Cr
        self.lf = lf
        self.lr = lr
        self.Iz = Iz
        self.I_axle = I_axle
        self.int = 0.0  
        self.m = m
        self.r = r

    def dynamics(self, Vx):

        Vx = max(abs(Vx), 0.01) 

        A = np.array([
            [(-2*self.Cf - 2*self.Cr) / (self.m * Vx),           0,
             -Vx - (2*self.Cf*self.lf - 2*self.Cr*self.lr) / (self.m*Vx)],
            [0,                                               0,   1],
            [(-(2*self.lf*self.Cf - 2*self.lr*self.Cr)) / (self.Iz*Vx), 0,
             -(2*self.lf**2*self.Cf + 2*self.lr**2*self.Cr) / (self.Iz*Vx)]
        ])

        B = np.array([
            2*self.Cf / self.m,
            0.0,
            2*self.lf*self.Cf / self.Iz
        ])
        return A, B
    
    def pid_yaw(self, yaw_error):
        self.I += yaw_error * self.dt  
        yaw_control = (
            self.Kp * yaw_error +
            self.Kd * (yaw_error - self.previous_error) / self.dt +
            self.I * self.Ki
        )
        self.previous_error = yaw_error
        return yaw_control
        
    def solver_vel(self, u):
        
        self.int += (u / self.I_axle) * self.dt/self.r
        return max(self.int, 0.01)  

    def solver_yaw(self, X_current, yaw_error, u, process_noise_std=0.0):

        Vx = self.solver_vel(u)
        A, B = self.dynamics(Vx)
        yaw_control = self.pid_yaw(yaw_error)

        X_next = self.dt * (A @ X_current) + self.dt * (B * yaw_control) + X_current

        if process_noise_std > 0.0:
            w = np.random.randn(len(X_next)) * process_noise_std
            X_next += w * X_next 

        return X_next