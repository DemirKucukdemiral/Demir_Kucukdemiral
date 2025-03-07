clear all; close all; clc;

global Tare_X Tare_Y Tare_Z specificGasConstant Area velocity

Taredata = readtable('ATP_yaw0_pitchtare_clean_elev_down.csv'); 
Taredata(end, :) = []; 
Taredata(5, :) = [];  


Tare_X = Taredata{:, 'Fx_D_'};
Tare_Y = Taredata{:, 'Fy_S_'};
Tare_Z = Taredata{:, 'Fz_L_'};


specificGasConstant = 287.05;
Area = 0.4;
velocity = 20;


Yaw0Data = readtable('ATP_yaw0_pitchsweep_clean_elev_down.csv');
Yaw5Data = readtable('ATP_yaw5_pitchsweep_clean_elev_down.csv');
Yaw10Data = readtable('ATP_yaw10_pitchsweep_clean_elev_down.csv');
Yaw15Data = readtable('ATP_yaw15_pitchsweep_clean_elev_down.csv');


dataPlotterBlock(Yaw0Data, 0);
dataPlotterBlock(Yaw5Data, 5);
dataPlotterBlock(Yaw10Data, 10);
dataPlotterBlock(Yaw15Data, 15);


function [] = dataPlotterBlock(data, yaw)
    global Tare_X Tare_Y Tare_Z specificGasConstant Area velocity

    yawRad = deg2rad(yaw);

    data(end, :) = [];

    avgTemp = mean(data{: , "Air_Temp__C_"} + 273.15, 'omitnan'); 
    avgPressure = mean(data{:, 'Baro_P_Pa_'}, 'omitnan');
    density = avgPressure / (avgTemp * specificGasConstant);


    q = 0.5 * Area * velocity^2 * density;

    F_x = -(data{: , 'Fx_D_'} - Tare_X);
    F_y =  (data{: , 'Fy_S_'} - Tare_Y);
    F_z = -(data{: , 'Fz_L_'} - Tare_Z);

    ForceArray = [F_x.'; F_y.'; F_z.'];  

    N = size(data, 1);
    pitchArray = data{: , 'Incidence'};
    
    R_cells = cell(N,1);

    for i = 1:N
        pitchRad = deg2rad(pitchArray(i));
        R_cells{i} = RotationMatrix(yawRad, pitchRad);
    end

    R_big = blkdiag(R_cells{:});

    ForceAll = ForceArray(:);      
    AeroForcesAll = R_big * ForceAll;  

    AeroForcesArray = reshape(AeroForcesAll, 3, N);

    CoefficientOfForces = AeroForcesArray / q;  


    figure;
    
    subplot(1, 3, 1)
    plot(pitchArray, CoefficientOfForces(1, :), 'o-');
    axis square
    set(gca, 'FontSize', 16);
    xlabel('Pitch Angle, ($\theta$) [deg]', 'Interpreter','latex');
    ylabel('Coefficient of Drag ($C_D$)', 'Interpreter','latex');
    title(sprintf('(a) Drag Coefficient Against Pitch Angle at Yaw = %.1f$^{\\circ}$', yaw), 'Interpreter', 'latex', 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
    grid on;

    subplot(1, 3, 2)
    plot(pitchArray, CoefficientOfForces(2, :), 'o-');
    axis square
    set(gca, 'FontSize', 16);
    xlabel('Pitch Angle, ($\theta$) [deg]', 'Interpreter','latex');
    ylabel('Coefficient of Side Force ($C_L$)', 'Interpreter','latex');
    title(sprintf('(b) Side Force Coefficient Against Pitch Angle at Yaw = %.1f$^{\\circ}$', yaw), 'Interpreter', 'latex', 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
    grid on;

    subplot(1, 3, 3)
    plot(pitchArray, CoefficientOfForces(3, :), 'o-');
    axis square
    set(gca, 'FontSize', 16);
    xlabel('Pitch Angle, ($\theta$) [deg]', 'Interpreter','latex');
    ylabel('Coefficient of Lift ($C_L$)', 'Interpreter','latex');
    title(sprintf('(c) Lift Coefficient Against Pitch Angle at Yaw = %.1f$^{\\circ}$', yaw), 'Interpreter', 'latex', 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
    grid on;
end


function R = RotationMatrix(yaw, pitch)
    alpha = pitch;
    beta  = -yaw;
    R = [-cos(beta)*cos(alpha),  -sin(beta),         -cos(beta)*sin(alpha);
         sin(beta)*cos(alpha),  -cos(beta),          sin(beta)*sin(alpha);
         sin(alpha),             0,                  -cos(alpha)
    ];
end
