clear all; close all; clc;


U = input('Give initial velocity: ');

y_axis = @(t, U, theta) U*sin(theta).*t - 0.5*9.81*t.^2;
x_axis = @(t, U, theta) U*cos(theta).*t;


figure;
hold on; 


for i = 5:5:85
    angle = deg2rad(i);
    
    flight_time = (2 * U * sin(angle)) / 9.81;
    
    t = linspace(0, flight_time, 100);
    
    y_values = y_axis(t, U, angle);
    x_values = x_axis(t, U, angle);
    
    plot(x_values, y_values);
end

xlabel('Horizontal Distance (m)');
ylabel('Vertical Distance (m)');
title('Projectile Motion for Different Angles');
grid on;
hold off; 
