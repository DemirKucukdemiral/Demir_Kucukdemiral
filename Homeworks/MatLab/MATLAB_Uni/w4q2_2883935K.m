clear all; close all; clc

boxmaker(5, 4); 

function [] = boxmaker(x, y)
    if (x <= 0 || mod(x, 1) ~= 0)
        fprintf('%.0f is not a positive integer.\n', x);
        return;
    end
    if (y <= 0 || mod(y, 1) ~= 0)
        fprintf('%.0f is not a positive integer.\n', y);
        return;
    end

    box_matrix = repmat(' ', y, x);
    
    box_matrix(:, 1) = '*';   
    box_matrix(:, x) = '*';   
    box_matrix(1, :) = '*';   
    box_matrix(y, :) = '*';   

    for i = 1:y
        fprintf('%s\n', box_matrix(i, :));
    end
end
