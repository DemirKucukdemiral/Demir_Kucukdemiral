clear all; close all; clc

product = 1;
number_of_trials = 0;
numbers = [];

while product <= 500
    num = input('please enter a positive number: '); 
    
    if num <= 0
        fprintf('Warning: non positive numbers are not allowed, please enter a positive number.\n');
        continue; 
    end
    
    % Update values
    number_of_trials = number_of_trials + 1;
    numbers(number_of_trials) = num; 
    product = product * num; 
    
    fprintf('%d : x = %.3f  , product = %.3f\n', number_of_trials, num, product);
end
fprintf('\nFinal Results:\n');
fprintf('Total numbers entered: %d', number_of_trials);
fprintf('    ');
fprintf('Final product: %.3f', product);
fprintf('   ');
fprintf('Numbers entered: ');
fprintf('%.3f ', numbers);
