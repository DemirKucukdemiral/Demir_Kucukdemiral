"""
Demir Kucukdemiral
2883935K

This program allows a user to call an object, Func(), allowing the 
computation of single integrals using the techniques:

    -Rectangular Rule
    -Trapazoidal rule

It also plots a function of the integration graphs vs the number of iterations

It also ensures that the integration is allowed to be computed even in the case of there 
being a mild singularity, therefore allowing the user to compute singular integrals easily.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

#Let us define the object which will contain the different integration methods
class Func():
    def __init__(self, func, a, b, eps):                      #initialising the parameters
        self.Eps = eps                                        #step size                       
        self.f = func                                         #interated function                
        self.a = a                                            #lower limit                    
        self.b = b                                            #upper limit                
        self.integartion_points = self.singularity_finder()   #points of integration found using the singularitry finder 
        

    #The singularity_finder function goes through every point along the function until it gives a ZeroDivisionError at which it saves the point just before and after this point so that while we are integrating we can split the integration into multiple parts to avoid the singularities                                                     
    def singularity_finder(self):
        try:                                                           #checks if the initial point is a singularity
            self.f(self.a)
            points = [self.a]    
            print(points)
        except (ZeroDivisionError, OverflowError, ValueError):
            points = [self.a + self.Eps]                               #if there is an error at this point, it adds the point just after this as the start position
            print(points)
                                                                    
        for i in range(1, int((self.b-self.a)/self.Eps)):              #this loop goes through every point in the function and checks if it is a singularity or not.
            try:
                self.f(self.a + i*self.Eps)
            except (ZeroDivisionError, OverflowError, ValueError):      #if there is a ZeroDivisonError, OverflowError or Value Error its going to know that this point is a singularity. To be able to go around it it sets limits just behind and after this point
                points.append(self.a + i*self.Eps - self.Eps)           #adding these points into the limits array
                points.append(self.a + i*self.Eps + self.Eps)
        points.append(self.b)   
        print(points)                                                  #adds the final value into the array, noted that this point should not be a singularity
        return points
    
    """
    The following 2 functions define how the numerical integration (sum) can be computed between two limits using rectangular and trapezoidal using their definition respectfully 
    """

    def integral1(self, a, b):                                     
        n = int((b - a) / self.Eps)                                #number of steps
        S = 0                                                      #initialises the sum
        for i in range(0, n):                                      #loop that goes through the function and calculates the size of rectangles and adds them into the sum
            S += self.f(a + i * self.Eps) * self.Eps         
        return S
    
    
    def integral2(self, a, b):
        n = int((b - a) / self.Eps) 
        S = 0.5 * self.Eps* (self.f(a) + self.f(b))      #adds the first and last value for integral so that the rest of the points can all be added in a loop
        for i in range(1, n):                            #treats rest of the points as a simple rectangular integral
            S += self.f(a + i * self.Eps)*self.Eps
        return S
    
    """
    The next two functions use the prevoiusly made definitions of rectangular and trapezoidal rules between two limits to compute between multiple sets of limits to avoid the singularities
    """
    def integralRect(self):
        S = 0
        for i in range(0, len(self.integartion_points)-1, 2):
            S += self.integral1(self.integartion_points[i], self.integartion_points[i+1])
        return S

    def integralTrap(self):
        S = 0
        for i in range(0, len(self.integartion_points)-1, 2):
            S += self.integral2(self.integartion_points[i], self.integartion_points[i+1])
        return S

    

"""
Plot the numerical approximations (Rectangular and Trapezoidal)
for the integral of `f` from `a` to `b`,
 as n (the number of iterations) goes from 2 to 100.
"""

#creating a functions to find the result at a much smaller epsilon 

def value(f, lowerLimit, upperLimit):
    eps = (upperLimit - lowerLimit) / 1000000
    integrator = Func(f, lowerLimit, upperLimit, eps)
    print("I=" , integrator.integralTrap())


def plotter(f, lower_limit, upper_limit):
    
    n_vals = range(1, 100)                              #creating x-axis
    rect_plotter = []                                   #defining arrays for trapezoidal and rectangular integration values
    trap_plotter = []
    

    for n in n_vals:                                    #computes and adds the values of integration for a given n and adds it onto ar array 
        eps = (upper_limit - lower_limit) / n
        integrator = Func(f, lower_limit, upper_limit, eps)

        rect_plotter.append(integrator.integralRect())
        trap_plotter.append(integrator.integralTrap())
        
#prints the values of integration using both methods after 100 iterations
    print('Final value:')
    print('With Trapezoidal Rule:' , trap_plotter[98])
    print('With Rectangular Rule:' , rect_plotter[98])
    
    plt.figure(figsize=(8, 6))                                                           #settingb figure size

    plt.plot(n_vals, rect_plotter, label="Rectangular", marker='o', markersize=3)        #defining the plot of rectangular and trapezoidal along with their appearance
    plt.plot(n_vals, trap_plotter, label="Trapezoidal", marker='x', markersize=3)
    plt.xlabel("Number of Subintervals (n)")
    plt.ylabel("Approximate Integral")
    plt.title("Numerical Integration Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(n_vals)

#defining the functions to be evaluated
def f1(x):
    return x

def f2(x):
    return math.sin(x)

def f3(x):
    return math.e**(-x)

functions = [f1, f2, f3]

#evaluating all in one loop 

for i in functions:
    plotter(i, 0, 1)


