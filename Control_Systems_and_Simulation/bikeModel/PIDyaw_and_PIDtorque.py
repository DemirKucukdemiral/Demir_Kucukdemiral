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
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

    def set_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def reset(self):
        self.error_array = np.array([0.0, 0.0])
        self.integral = 0.0

    def PID(self, error):
        e_prev = self.error_array[-1]
        de = (error - e_prev) / self.dt
        self.integral += error * self.dt
        u = self.Kp * error + self.Ki * self.integral + self.Kd * de

        self.error_array = np.append(self.error_array, error)
        if len(self.error_array) > 2:
            self.error_array = self.error_array[-2:]
        return u


class Bike:

    def __init__(self, Cf, Cr, m, lr, lf, Iz, dt, process_noise_std=0.0):
        self.Cf = Cf
        self.Cr = Cr
        self.m  = m
        self.lr = lr
        self.lf = lf
        self.Iz = Iz
        self.dt = dt
        self.process_noise_std = process_noise_std
        self.wheel_radius = 0.2
        

    def dynamics(self, Vx):
   
        if abs(Vx) < 0.01:
            Vx = 0.01

        A = np.array([
            [(-2*self.Cf - 2*self.Cr)/(self.m * Vx),           0,
             -Vx - (2*self.Cf*self.lf - 2*self.Cr*self.lr)/(self.m*Vx)],
            [0,                                               0,   1],
            [(-(2*self.lf*self.Cf - 2*self.lr*self.Cr))/(self.Iz*Vx), 0,
             -(2*self.lf**2*self.Cf + 2*self.lr**2*self.Cr)/(self.Iz*Vx)]
        ])

        B = np.array([
            2*self.Cf/self.m,
            0.0,
            2*self.lf*self.Cf/self.Iz
        ])
        return A, B

    def solver_pid(
        self,
        Vx_init,
        X_init,          # [x1, x2=Yaw, x3=YawRate]
        X_ref_list,      # multiple references (yaw angle, yaw rate)
        intervals_yaw,   # times to switch yaw references
        Vx_refs,         # velocity references
        intervals_vx,    # times to switch velocity references
        T,
        pid_velocity,    # PID for velocity
        pid_yaw          # PID for yaw control
    ):
        
        dt = self.dt
        steps = int(T / dt)
        time_array = np.linspace(0, T, steps+1)

        # Data logs
        Yaw_array     = np.zeros(steps+1)
        Yaw_rate_array= np.zeros(steps+1)
        Velocity      = np.zeros(steps+1)
        Delta_yaw     = np.zeros(steps+1)

        X_current = X_init.copy()
        Vx = Vx_init

        Yaw_array[0]      = X_current[1]
        Yaw_rate_array[0] = X_current[2]
        Velocity[0]       = Vx
        Delta_yaw[0]      = 0.0  
        yaw_ref_index = 0
        vx_ref_index  = 0

        for i in range(steps):
            t_now = (i+1)*dt

            if yaw_ref_index < len(X_ref_list) - 1:
                if t_now >= intervals_yaw[yaw_ref_index]:
                    yaw_ref_index += 1
            current_yaw_ref = X_ref_list[yaw_ref_index]

            if vx_ref_index < len(Vx_refs) - 1:
                if t_now >= intervals_vx[vx_ref_index]:
                    vx_ref_index += 1
            current_Vx_ref = Vx_refs[vx_ref_index]

            velocity_error = current_Vx_ref - Vx
            u_vel = pid_velocity.PID(velocity_error)
            Vx += u_vel * dt

                        
            T_engine = pid_velocity.PID(velocity_error)
            F_drive = T_engine / self.wheel_radius
            
            a_long = (F_drive) / m
            Vx += a_long * dt

            A, B = self.dynamics(Vx)

            yaw_error = current_yaw_ref[1] - X_current[1]
            delta_yaw_in = pid_yaw.PID(yaw_error)

            max_steer = np.radians(30.0)
            delta_yaw_in = np.clip(delta_yaw_in, -max_steer, max_steer)

            X_dot = A @ X_current + B * delta_yaw_in
            X_current += dt * X_dot

            if self.process_noise_std > 0.0:
                w = np.random.randn(len(X_current)) * self.process_noise_std
                X_current += w

            Yaw_array[i+1]      = X_current[1]
            Yaw_rate_array[i+1] = X_current[2]
            Velocity[i+1]       = Vx
            Delta_yaw[i+1]      = delta_yaw_in

        plt.figure(figsize=(10,5))
       
        plt.subplot(2,1,1)
        plt.plot(time_array, Yaw_array, label='Yaw (PID)')

        yaw_ref_array = np.zeros_like(time_array)
        for k in range(steps+1):
            t_temp = time_array[k]
            idx = 0
            while idx < len(intervals_yaw) and t_temp >= intervals_yaw[idx]:
                idx += 1
            if idx >= len(X_ref_list):
                idx = len(X_ref_list)-1
            yaw_ref_array[k] = X_ref_list[idx][1]
        plt.plot(time_array, yaw_ref_array, 'k--', label='Yaw Ref')

        plt.ylabel("Yaw [rad]")
        plt.legend()
        plt.grid()


        plt.subplot(2,1,2)
        plt.plot(time_array, Yaw_rate_array, label='Yaw Rate (PID)')

        
        yawrate_ref_array = np.zeros_like(time_array)
        for k in range(steps+1):
            t_temp = time_array[k]
            idx = 0
            while idx < len(intervals_yaw) and t_temp >= intervals_yaw[idx]:
                idx += 1
            if idx >= len(X_ref_list):
                idx = len(X_ref_list)-1
            yawrate_ref_array[k] = X_ref_list[idx][2]
        plt.plot(time_array, yawrate_ref_array, 'k--', label='Yaw Rate Ref')

        plt.ylabel("Yaw Rate [rad/s]")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


        plt.figure()
        plt.plot(time_array, Velocity, label="Velocity (PID)")
        vx_ref_array = np.zeros_like(time_array)
        for k in range(steps+1):
            t_temp = time_array[k]
            idx = 0
            while idx < len(intervals_vx) and t_temp >= intervals_vx[idx]:
                idx += 1
            if idx >= len(Vx_refs):
                idx = len(Vx_refs)-1
            vx_ref_array[k] = Vx_refs[idx]
        plt.plot(time_array, vx_ref_array, 'k--', label="Vx Ref")
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity [m/s]")
        plt.legend()
        plt.grid()
        plt.show()


        plt.figure()
        plt.plot(time_array, Delta_yaw, label="Steering (PID)")
        plt.axhline(y= max_steer, color='r', linestyle='--', label="±30 deg limit")
        plt.axhline(y=-max_steer, color='r', linestyle='--')
        plt.xlabel("Time [s]")
        plt.ylabel("Steering Angle [rad]")
        plt.title("Steering Input (Yaw PID)")
        plt.grid()
        plt.legend()
        plt.show()



if __name__ == "__main__":
    
    Cf = 19000
    Cr = 33000
    m  = 1500
    lr = 1.6
    lf = 1.2
    Iz = 3000

    dt = 0.01
    process_noise_std = 0.01 

    bike = Bike(Cf, Cr, m, lr, lf, Iz, dt, process_noise_std)

    pid_velocity = PIDController(dt)
    pid_velocity.set_gains(Kp=20.0, Ki=0.1, Kd=0.5)  

    pid_yaw = PIDController(dt)
    pid_yaw.set_gains(Kp=5.0, Ki=10.0, Kd=1.0)       

  
    X_init = np.array([0.0, 0.1, 0.05])
    Vx_init = 15.0

    X_ref_list = [
        np.array([0.0, np.radians(30.0), 0.0]),
        np.array([0.0, np.radians(50.0), 0.0]),
        np.array([0.0, np.radians(10.0), 0.0])
    ]
    intervals_yaw = [20.0, 40.0]

  
    Vx_refs = [20.0, 30.0, 10.0]
    intervals_vx = [15.0, 40.0]

    T = 60.0 


    bike.solver_pid(
        Vx_init       = Vx_init,
        X_init        = X_init,
        X_ref_list    = X_ref_list,
        intervals_yaw = intervals_yaw,
        Vx_refs       = Vx_refs,
        intervals_vx  = intervals_vx,
        T             = T,
        pid_velocity  = pid_velocity,
        pid_yaw       = pid_yaw
    )