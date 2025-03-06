clear all; close all; clc;

n = input('Please enter n: ');


if n < 0 || mod(n,1) ~= 0
    disp('Error: Please enter a positive integer.');
    return;
end


fprintf('\nn =\tFib(n) =\tRatio\n');

fib_array = Fib(n);

for i = 0:n
    if i == 0
        fprintf('%d\t%d\t\t(undefined)\n', i, fib_array(i+1));
    elseif i == 1
        fprintf('%d\t%d\t\t(undefined)\n', i, fib_array(i+1));
    else
        ratio = fib_array(i+1) / fib_array(i);
        fprintf('%d\t%d\t\t%.6f\n', i, fib_array(i+1), ratio);
    end
end

function fib_array = Fib(n)
   
    fib_array = zeros(1, n+1); 
    fib_array(1) = 0; 
    if n >= 1
        fib_array(2) = 1; 
    end

    for i = 3:n+1
        fib_array(i) = fib_array(i-1) + fib_array(i-2);
    end
end


