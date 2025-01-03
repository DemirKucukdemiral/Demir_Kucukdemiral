import pygame
import numpy as np

"""
This code provides a solution to artificial potential field algorithm problem APF, using a logarithmic repulsive term

-Demir Kucukdemiral
-2883935K@student.gla.ac.uk
"""

# Initialize Pygame
pygame.init()
win_width, win_height = 1000, 800
win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Robot Potential Field Simulation with Log Barrier")
width = 20

# Initial positions
x0, y0 = 300.0, 280.0  # Robot's initial position
xf0, yf0 = 1000.0, 300.0  # Goal position
x = np.array([x0, y0], dtype=np.float64)
xf = np.array([xf0, yf0], dtype=np.float64)
rad = 50

# Obstacle positions
xO = np.array([[420, 240], [500, 350], [700, 300], [860, 320]], dtype=np.float64)
n_obst = len(xO)

Klog = 40     # Coefficient for log barrier term
psi = 3       # Coefficient for attractive term
d_goal = 1    # Distance at which attractive potential becomes quadratic
alpha = 10    # Step size coefficient
Q_away = 100  # Distance at which the log barrier becomes active

# Colors
BACKGROUND_COLOR = (25, 25, 30)
ROBOT_COLOR = (255, 255, 255)
GOAL_COLOR = (255, 50, 50)
OBSTACLE_COLOR = (255, 255, 255)
PATH_COLOR = (0, 255, 0)


clock = pygame.time.Clock()

#path array
path = []

# Attraction potential function
def U_att(x, xf):
    d = np.linalg.norm(x - xf)
    if d <= d_goal:
        return 0.5 * psi * d**2
    else:
        return psi * d_goal * d - 0.5 * psi * d_goal**2

#Gradient of attractive potential
def grad_U_att(x, xf):                     
    d = np.linalg.norm(x - xf)
    if d == 0:
        return np.array([0.0, 0.0])
    elif d <= d_goal:
        return psi * (x - xf)
    else:
        return psi * d_goal * (x - xf) / d

#Repulsive, log term
def U_rep_log(x, xO):                       
    d = np.linalg.norm(x - xO)
    if d <= Q_away:
        return -Klog * np.log(d - rad)
    else:
        return 0.0

 #derivetive of the log term
def grad_U_rep_log(x, xO):                 
    d = np.linalg.norm(x - xO)
    if d <= Q_away and d > rad:
        return -Klog * (x - xO) / (d * (d - rad))
    else:
        return np.array([0.0, 0.0])


def compute_gradient(x, xf, xO_list):           #magnitude of the gradient terms
    grad = grad_U_att(x, xf)
    for xO in xO_list:
        grad += grad_U_rep_log(x, xO)
    return grad

# Main loop
run = True
while run:
    clock.tick(60)  # Limit to 30 frames per second

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Compute gradient and update position
    grad = compute_gradient(x, xf, xO)
    grad_norm = np.linalg.norm(grad)

    if grad_norm > 0:
        grad = grad / grad_norm         # Normalize the gradient
        x = x - alpha * grad * 0.5      

        # Keep the robot within window bounds
        x[0] = np.clip(x[0], 0, win_width - width)
        x[1] = np.clip(x[1], 0, win_height - width)

    path.append((int(x[0] + width / 2), int(x[1] + width / 2)))                             #path array


    #This part is about drawing all the parts to be displayed
    win.fill(BACKGROUND_COLOR)
    for i in range(n_obst):
        pygame.draw.circle(win, OBSTACLE_COLOR, (int(xO[i][0]), int(xO[i][1])), 50)         #drawing obstacles
    if len(path) > 1:
        pygame.draw.lines(win, PATH_COLOR, False, path, 2)                                  #drawing the path of the robot
    pygame.draw.rect(win, ROBOT_COLOR, (int(x[0]), int(x[1]), width // 2, width // 2))      #drawing the robot itself
    pygame.draw.rect(win, GOAL_COLOR, (int(xf[0]), int(xf[1]), width, width))               #drawing the goal
    pygame.display.update()

pygame.quit()
