import numpy as np
import control
import pygame
import math


class Bike:
    def __init__(self, Cf, Cr, m, lr, lf, Iz, Q, R, dt):
        self.Cf = Cf
        self.Cr = Cr
        self.m = m
        self.lr = lr
        self.lf = lf
        self.Iz = Iz
        self.Q = Q
        self.R = R
        self.dt = dt

    def dynamics(self, Vx):
     
        A = np.array([
            [(-2*self.Cf - 2*self.Cr)/(self.m*Vx), 0,
             -Vx - (2*self.Cf*self.lf - 2*self.Cr*self.lr)/(self.m*Vx)],
            [0, 0, 1],
            [-(2*self.lf*self.Cf - 2*self.lr*self.Cr)/(self.Iz*Vx), 0,
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
        Ad = np.eye(3) + self.dt * (A - B @ K)
        Bd = self.dt * (B @ K)
        return Ad @ X + Bd @ X_ref

class APFPathFinder:
   
    def __init__(self, pos, obstacles, goal):
        self.pos = np.array(pos, dtype=float)
        self.obstacles = [np.array(o, dtype=float) for o in obstacles]
        self.goal = np.array(goal, dtype=float)

        self.psi = 3
        self.d_goal = 0.5
        self.alpha = 10
        self.Q_away = 100   
        self.mu = -300000
        self.gamma = 2
        self.damping = 0.9

    def U_att(self):
        d = np.linalg.norm(self.pos - self.goal)
        if d <= self.d_goal:
            return 0.5*self.psi*(d**2)
        else:
            return self.psi*self.d_goal*d - 0.5*self.psi*(self.d_goal**2)

    def grad_U_att(self):
        d = np.linalg.norm(self.pos - self.goal)
        if d == 0:
            return np.array([0.0, 0.0])
        elif d <= self.d_goal:
            return self.psi*(self.pos - self.goal)
        else:
            return self.psi*self.d_goal*(self.pos - self.goal)/d

    def U_rep(self, obs):
        d = np.linalg.norm(self.pos - obs)
        if d < self.Q_away:
            return (1/self.gamma)*self.mu*(1/d - 1/self.Q_away)**self.gamma
        else:
            return 0.0

    def grad_U_rep(self, obs):
        d = np.linalg.norm(self.pos - obs)
        if d == 0 or d > self.Q_away:
            return np.array([0.0, 0.0])
        else:
            coeff = self.mu*(1/d - 1/self.Q_away)**(self.gamma-1)*(1/d**2)
            return coeff*(self.pos - obs)/d

    def compute_gradient(self):
        grad = self.grad_U_att()
        for obs in self.obstacles:
            grad += self.grad_U_rep(obs)
        return grad

    def step(self):
        grad = self.compute_gradient()
        g_norm = np.linalg.norm(grad)
        if g_norm > 1e-9:
            step_size = self.alpha*g_norm
            self.pos -= (grad/g_norm)*step_size*self.damping

    def goal_reached(self, threshold=8.0):
        return np.linalg.norm(self.pos - self.goal) < threshold


def main():

    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("APF + LQR in one window")
    clock = pygame.time.Clock()

    # Colors
    bg_color = (25, 25, 30)
    obs_color = (255, 255, 255)
    robot_color = (0, 200, 255)
    path_color = (128, 128, 128) 
    goal_color = (255, 50, 50)
    waypoint_color = (255, 215, 0)

    APF_MODE = 0
    LQR_MODE = 1
    mode = APF_MODE

    obstacles = [(400, 350), (600, 350), (500, 300), (200, 250)]
    start_pos = (100, 180)
    goal_pos = (700, 380)
    apf = APFPathFinder(start_pos, obstacles, goal_pos)
    apf_path = [apf.pos.copy()]


    waypoints = []
    apf_done = False

    # LQR / Bike Setup
    Cf, Cr = 19000, 33000
    m = 1500
    lr, lf = 1.6, 1.2
    Iz = 3000
    Q = np.diag([1, 40, 5])
    R = np.array([[5]])
    dt = 0.01
    bike = Bike(Cf, Cr, m, lr, lf, Iz, Q, R, dt)

    # LQR states
    vx = 30  
    X_bike = np.array([0.0, 0.0, 0.0])  
    car_x, car_y = start_pos
    lqr_path = [(car_x, car_y)]
    waypoint_index = 0

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(bg_color)


        for obs in obstacles:
            pygame.draw.circle(screen, obs_color, (int(obs[0]), int(obs[1])), 30)

       
        if mode == APF_MODE:
            if not apf.goal_reached():
                apf.step()
                apf_path.append(apf.pos.copy())
            else:
           
                if not apf_done:
                    apf_done = True
                  
                    waypoints = apf_path[::2]
                    waypoints.append(apf.pos.copy())
                    mode = LQR_MODE
    
                    car_x, car_y = waypoints[0]
                    lqr_path = [(car_x, car_y)]
                    waypoint_index = 1

          
            if len(apf_path) > 1:
                pygame.draw.lines(
                    screen, path_color, False,
                    [(int(p[0]), int(p[1])) for p in apf_path], 2
                )
   
            pygame.draw.circle(screen, robot_color,
                               (int(apf.pos[0]), int(apf.pos[1])), 7)
        
            pygame.draw.rect(screen, goal_color,
                             (int(goal_pos[0]) - 10, int(goal_pos[1]) - 10, 20, 20))


        elif mode == LQR_MODE:
   
            if len(apf_path) > 1:
                pygame.draw.lines(
                    screen, path_color, False,
                    [(int(p[0]), int(p[1])) for p in apf_path], 2
                )
       
            for wpt in waypoints:
                pygame.draw.circle(screen, waypoint_color,
                                   (int(wpt[0]), int(wpt[1])), 5)


            if waypoint_index < len(waypoints):
                wx, wy = waypoints[waypoint_index]
                dx, dy = wx - car_x, wy - car_y
                dist = math.hypot(dx, dy)
                if dist < 15:
                    waypoint_index += 1
                else:
                 
                    yaw_ref = math.atan2(dy, dx)
                    X_ref = np.array([0.0, yaw_ref, 0.0])
                    X_bike = bike.next_step(vx, X_bike, X_ref)
                    v_y, yaw, r = X_bike

                    # update global
                    car_x_dot = vx*math.cos(yaw) - v_y*math.sin(yaw)
                    car_y_dot = vx*math.sin(yaw) + v_y*math.cos(yaw)
                    car_x += car_x_dot*dt
                    car_y += car_y_dot*dt
                    lqr_path.append((car_x, car_y))
            else:
                pass

       
            if len(lqr_path) > 1:
                pygame.draw.lines(
                    screen, (0, 255, 255), False,
                    [(int(px), int(py)) for (px, py) in lqr_path], 2
                )
     
            pygame.draw.circle(screen, robot_color, (int(car_x), int(car_y)), 7)
    
            pygame.draw.rect(screen, goal_color,
                             (int(goal_pos[0]) - 10, int(goal_pos[1]) - 10, 20, 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
