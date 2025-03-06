clear all; close all; clc

[error_msg, tip, total] = charger(1, 10); 

disp(error_msg); 
disp(['Tip: ', num2str(tip)]);
disp(['Total: ', num2str(total)]);

function [error_msg, tip, total] = charger(service, price)
    possible_ratings = [1, 2, 3];
    error_msg = ''; 
    tip = 0; 
    total = 0; 

    if ~ismember(service, possible_ratings)
        error_msg = sprintf('%d is not an available score, please enter: 1, 2, or 3', service);
    else 
        percentage = service * 5 / 100;
        tip = percentage * price;
        total = price + tip;
    end
end
