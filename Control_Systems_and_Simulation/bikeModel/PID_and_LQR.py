import numpy as np
import matplotlib.pyplot as plt
import control 

"""
States:
    X[0]: lateral velocity
    x[1]:yaw
    x[2]:yaw rate
Control inputs:
    u = steering angle
"""


class PIDController:
    def __init__(self, dt):
        self.dt = dt
        self.error_array = np.array([0.0, 0.0])
        self.integral = 0.0

    def PID(self, Kp, Ki, Kd):
        e = self.error_array[-1]
        e_prev = self.error_array[-2]
        de = (e - e_prev) / self.dt  
        self.integral += e * self.dt  
        return Kp * e + Ki * self.integral + Kd * de


class Bike():
    def __init__(self, Cf, Cr, m, lr, lf, Iz, Q, R, t, process_noise_std):
        self.Cf = Cf
        self.Cr = Cr
        self.m = m
        self.lr = lr
        self.lf = lf
        self.Iz = Iz
        self.Q = Q  
        self.R = R  
        self.t = t  
        self.process_noise_std = process_noise_std

    def dynamics(self, Vx):
      
        A = np.array([
            [(-2*self.Cf - 2*self.Cr)/(self.m * Vx),           0,
             -Vx - (2*self.Cf*self.lf - 2*self.Cr*self.lr)/(self.m*Vx)],
            [0,                                               0,   1],
            [(-(2*self.lf*self.Cf - 2*self.lr*self.Cr))/(self.Iz*Vx), 0,
             -(2*self.lf**2*self.Cf + 2*self.lr**2*self.Cr)/(self.Iz*Vx)]
        ])

        B = np.array([
            [2*self.Cf/self.m],
            [0],
            [2*self.lf*self.Cf/self.Iz]
        ])
        return A, B

    def compute_lqr_gain(self, A, B):
        K, _, _ = control.lqr(A, B, self.Q, self.R)
        return K

    def next_step(self, Vx, X, X_ref):
        A, B = self.dynamics(Vx)
        K = self.compute_lqr_gain(A, B)
        u_raw = -K @ (X - X_ref)  # Or however you form your LQR control
        
        # 1) Saturate the steering angle
        steering_angle_max = 30.0 * np.pi/180.0  # ±30 deg
        u_clamped = np.clip(u_raw, -steering_angle_max, steering_angle_max)
        
        # 2) Now apply the clamped steering input in the continuous-time dynamics
        # (If B has dimension 3x1, then the input is just that single steering angle.)
        X_dot = A @ X + B @ u_clamped
        X_new = X + self.t * X_dot
        if self.process_noise_std > 0.0:
             w = np.random.randn(len(X)) * self.process_noise_std
        X_new = X_new + w*X_new
        return X_new

    def solver(
        self, 
        Vx_init,          # initial velocity
        X_current,        # initial lateral states
        X_ref,            # list of lateral references (e.g. [ [0, yaw1, yawrate1], [0,yaw2,...], ... ])
        intervals_yaw,    # switch times for yaw references
        Vx_refs,          # list of velocity references (e.g. [20, 30, 10])
        intervals_vx,     # switch times for velocity references
        T,                # total sim time
        pid_controller
    ):
        steps = int(T / self.t)

        time_array = np.zeros(steps+1)
        Plotter_yaw = np.zeros(steps+1)
        Plotter_yaw_rate = np.zeros(steps+1)
        Velocity = np.zeros(steps+1)

        time_array[0] = 0.0
        Plotter_yaw[0] = X_current[1]
        Plotter_yaw_rate[0] = X_current[2]
        Vx = Vx_init
        Velocity[0] = Vx

        yaw_ref_index = 0
        v_ref_index = 0

        for i in range(steps):
            t_now = (i+1)*self.t
            time_array[i+1] = t_now

            if yaw_ref_index < len(X_ref) - 1:  
                if t_now >= intervals_yaw[yaw_ref_index]:
                    yaw_ref_index += 1
            current_yaw_ref = X_ref[yaw_ref_index]  

            if v_ref_index < len(Vx_refs) - 1:  
                if t_now >= intervals_vx[v_ref_index]:
                    v_ref_index += 1
            current_Vx_ref = Vx_refs[v_ref_index]

            velocity_error = current_Vx_ref - Vx
            pid_controller.error_array = np.append(pid_controller.error_array, velocity_error)
            
            if len(pid_controller.error_array) > 2:
                pid_controller.error_array = pid_controller.error_array[-2:]

            Vx += pid_controller.PID(20, 0.1, 0.5) * self.t

            X_current = self.next_step(Vx, X_current, current_yaw_ref)

            Plotter_yaw[i+1]      = X_current[1]
            Plotter_yaw_rate[i+1] = X_current[2]
            Velocity[i+1]         = Vx

        plt.figure(figsize=(10,4))
        plt.plot(time_array, Plotter_yaw, label="Yaw Angle (x2)")
        plt.plot(time_array, Plotter_yaw_rate, label="Yaw Rate (x3)")

        yaw_ref_array      = np.zeros_like(time_array)
        yaw_rate_ref_array = np.zeros_like(time_array)

        for i in range(steps+1):
            tnow = time_array[i]
         
            idx = 0
            while idx < len(intervals_yaw) and tnow >= intervals_yaw[idx]:
                idx += 1
            if idx >= len(X_ref):
                idx = len(X_ref)-1
            yaw_ref_array[i]      = X_ref[idx][1]
            yaw_rate_ref_array[i] = X_ref[idx][2]

        plt.plot(time_array, yaw_ref_array, '--', label="Yaw Angle Ref")
        plt.plot(time_array, yaw_rate_ref_array, '--', label="Yaw Rate Ref")
        plt.xlabel("Time [s]")
        plt.ylabel("Yaw, Yaw Rate")
        plt.title("Yaw Evolution with LQR")
        plt.grid(); plt.legend(); plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10,4))
        plt.plot(time_array, Velocity, label="Vx")

        vx_ref_array = np.zeros_like(time_array)
        for i in range(steps+1):
            tnow = time_array[i]
            idx = 0
            while idx < len(intervals_vx) and tnow >= intervals_vx[idx]:
                idx += 1
            if idx >= len(Vx_refs):
                idx = len(Vx_refs)-1
            vx_ref_array[i] = Vx_refs[idx]

        plt.plot(time_array, vx_ref_array, '--', label="Vx Ref")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.title("Velocity Evolution with PID Control")
        plt.grid(); plt.legend(); plt.tight_layout()
        plt.show()


Cf = 19000
Cr = 33000
m  = 1500
lr = 1.6
lf = 1.2
Iz = 3000


Q = np.diag([1, 50, 5])     
R = np.array([[ 50 ]])         


dt = 0.01
distribution_percentage = 0.01

#initalisation
bike = Bike(Cf, Cr, m, lr, lf, Iz, Q, R, dt, distribution_percentage)
pid_controller = PIDController(dt)


Vx_init    = 0                       
X_current  = np.array([0.0, 0.0, 0.05]) 


X_ref = [
    np.array([0.0,  30.0*np.pi/180.0, 0.0]),  
    np.array([0.0,  50.0*np.pi/180.0, 0.0]),  
    np.array([0.0,  10.0*np.pi/180.0, 0.0])   
]
intervals = [20, 40]  
T = 60  
Vx_refs= [20.0, 30.0, 10.0]
intervals_vx = [15.0, 40.0]  
intervals_yaw = [20.0, 40.0]


bike.solver(
    Vx_init     = 15.0,
    X_current   = np.array([0.0,  0.1,  0.01]), 
    X_ref       = X_ref,
    intervals_yaw = intervals_yaw,
    Vx_refs     = Vx_refs,
    intervals_vx = intervals_vx,
    T           = 60.0,
    pid_controller = pid_controller
)