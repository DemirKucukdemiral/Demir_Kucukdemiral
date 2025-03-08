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


CoefficientForces_Yaw0 = dataPlotterBlock(Yaw0Data, 0);
CoefficientForces_Yaw5 = dataPlotterBlock(Yaw5Data, 5);
CoefficientForces_Yaw10 = dataPlotterBlock(Yaw10Data, 10);
CoefficientForces_Yaw15 = dataPlotterBlock(Yaw15Data, 15);

yaw_angles = [0, 5, 10, 15];

pitchValues = Yaw0Data{:,'Incidence'};  
pitchValues(end) = [];

D = [CoefficientForces_Yaw0(1,:);
     CoefficientForces_Yaw5(1,:);
     CoefficientForces_Yaw10(1,:);
     CoefficientForces_Yaw15(1,:)];

S = [CoefficientForces_Yaw0(2,:);
     CoefficientForces_Yaw5(2,:);
     CoefficientForces_Yaw10(2,:);
     CoefficientForces_Yaw15(2,:)];

L = [CoefficientForces_Yaw0(3,:);
     CoefficientForces_Yaw5(3,:);
     CoefficientForces_Yaw10(3,:);
     CoefficientForces_Yaw15(3,:)];

[Pitch, Yaw] = meshgrid(pitchValues, yaw_angles);

figure;
subplot(1, 3, 1)
surf(Pitch, Yaw, D)
axis square
set(gca, 'FontSize', 25);
xlabel('Pitch Angle $\theta$ [deg]', Interpreter='latex')
ylabel('Yaw Angle $\psi$ [deg]', Interpreter='latex')
zlabel('Drag Coefficient, \(C_D\)', 'Interpreter', 'latex')
title('Drag Coefficient vs. Pitch and Yaw', Interpreter='latex')
grid on;

subplot(1, 3, 2)
surf(Pitch, Yaw, S)
axis square
set(gca, 'FontSize', 25);
xlabel('Pitch Angle $\theta$ [deg]', Interpreter='latex')
ylabel('Yaw Angle $\psi$ [deg]', Interpreter='latex')
zlabel('Side Force Coefficient, \(C_S\)', 'Interpreter', 'latex')
title('Side Force Coefficient vs. Pitch and Yaw', Interpreter='latex')
grid on;

subplot(1, 3, 3)
surf(Pitch, Yaw, L)
axis square
set(gca, 'FontSize', 25);
xlabel('Pitch Angle $\theta$ [deg]', Interpreter='latex')
ylabel('Yaw Angle $\psi$ [deg]', Interpreter='latex')
zlabel('Lift Coefficient, \(C_L\)', 'Interpreter', 'latex')
title('Lift Coefficient vs. Pitch and Yaw', Interpreter='latex')
grid on;


function CoefficientOfForces = dataPlotterBlock(dataIn, yaw)
    global Tare_X Tare_Y Tare_Z specificGasConstant Area velocity

    yawRad = deg2rad(yaw);
    data = dataIn;
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
    set(gca, 'FontSize', 25);
    xlabel('Pitch Angle, ($\theta$) [deg]', 'Interpreter', 'latex');
    ylabel('Coefficient of Drag ($C_D$)', 'Interpreter', 'latex');
    title(sprintf('(a) $C_D$ vs $\\theta$ at $\\psi$ = %.1f$^{\\circ}$', yaw), ...
        'Interpreter', 'latex', 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
    grid on;
    
    subplot(1, 3, 2)
    plot(pitchArray, CoefficientOfForces(2, :), 'o-');
    axis square
    set(gca, 'FontSize', 25);
    xlabel('Pitch Angle, ($\theta$) [deg]', 'Interpreter', 'latex');
    ylabel('Coefficient of Side Force ($C_L$)', 'Interpreter', 'latex');
    title(sprintf('(b) $C_S$ vs $\\theta$ at $\\psi$ = %.1f$^{\\circ}$', yaw), ...
        'Interpreter', 'latex', 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
    grid on;
    
    subplot(1, 3, 3)
    plot(pitchArray, CoefficientOfForces(3, :), 'o-');
    axis square
    set(gca, 'FontSize', 25);
    xlabel('Pitch Angle, ($\theta$) [deg]', 'Interpreter', 'latex');
    ylabel('Coefficient of Lift ($C_L$)', 'Interpreter', 'latex');
    title(sprintf('(c) $C_L$ vs $\\theta$ at $\\psi$ = %.1f$^{\\circ}$', yaw), ...
        'Interpreter', 'latex', 'FontWeight', 'bold', 'VerticalAlignment', 'bottom');
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
