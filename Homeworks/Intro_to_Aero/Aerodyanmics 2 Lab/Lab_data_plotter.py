import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

class pressureAnalysis():
    def __init__(self, data):
        self.data = data
        self.pressure_atm = 0.260  # Pa
        self.air_density = 1.225   # kg/m^3
        self.velocity_mean = np.mean(pd.to_numeric(self.data.iloc[5:, 47]))

        self.angles_to_measure = [0, 5, 10, 15, 20]

        self.x_axis = np.linspace(-6, 20, 14)
        
        self.pressure_data = data.iloc[4:, 12:32].reset_index(drop=True)

        self.pressure_data = self.pressure_data.apply(pd.to_numeric, errors='coerce')
        
        self.pressure_data.to_csv("pressure_data.csv", index=False)

        self.pressure_data.iloc[:, 7] = (self.pressure_data.iloc[: , 9] + self.pressure_data.iloc[:, 5])/2
        self.pressure_data.iloc[:, 8] = (self.pressure_data.iloc[: , 10] + self.pressure_data.iloc[:, 6])/2
        self.pressure_data.iloc[:, 16] = (self.pressure_data.iloc[: , 14] + self.pressure_data.iloc[:, 18])/2

        self.chord = 150  # mm
        
        self.upper_surface_pressures = self.pressure_data.iloc[:, 1::2]
        self.lower_surface_pressures = self.pressure_data.iloc[:, ::2]

        self.upper_surface_pressures_matrix = self.upper_surface_pressures.to_numpy()
        self.lower_surface_pressures_matrix = self.lower_surface_pressures.to_numpy()
    
        self.lowerDistances = np.array([0.76, 3.81, 11.43, 19.05, 38, 62, 80.77, 101.35, 121.92, 137.16])
        self.upperDistances = np.array([1.52, 7.62, 12.24, 22.86, 41.15, 59.44, 77.73, 96.02, 114.30, 129.54])
        
        self.Cp_upper = self.coefficientPressure(self.upper_surface_pressures_matrix) * 1000
        self.Cp_lower = self.coefficientPressure(self.lower_surface_pressures_matrix) *1000
        
        self.CL = self.trapezoidal_rule(self.Cp_upper, self.upperDistances, self.Cp_lower, self.lowerDistances)

        self.reynolds_number = 1.225 * self.velocity_mean * self.chord / (1.81e-5*1000)

        print(f"Reynolds number: {self.reynolds_number.round(2)}")
        

        


        data_naca0012 = np.loadtxt("naca0012_experiment_re_200000.txt")  

        
        self.angles_for_actual_experiment = data_naca0012[:, 0] 
        self.lift_cofficient_for_actual_experiment = data_naca0012[:, 1]



        print(len(self.Cp_upper[0]))
        plt.plot(self.upperDistances/self.chord, self.Cp_upper[7], label="Upper Surface")
        plt.plot(self.lowerDistances/self.chord, self.Cp_lower[7], label="Lower Surface")
        plt.xlabel("Chordwise Position (x/c)")
        plt.ylabel("Coefficient of Pressure")
        plt.title("Coefficient of Pressure vs Chordwise Position")
        plt.legend()
        plt.show()


    
    def coefficientPressure(self, pressure_to_plot):
        ones_matrix = np.ones_like(pressure_to_plot)
        print(pressure_to_plot)
        pressure_coefficient_matrix = (pressure_to_plot - self.pressure_atm * ones_matrix) / (0.5 * self.air_density * self.velocity_mean**2)
        return pressure_coefficient_matrix
    
    def trapezoidal_rule(self, matrix_1, distances_1, matrix_2, distances_2):
        distances_1 = np.array(distances_1) / self.chord
        dx_1 = distances_1.copy()
        for i in range(1, len(dx_1)):
            dx_1[i] = dx_1[i] - dx_1[i-1]

        self.upper_axis = dx_1
        
        dx_1[0] /= 2
        dx_1[-1] /= 2
        intMatrix_1 = matrix_1 @ dx_1
        
        distances_2 = np.array(distances_2) / self.chord
        dx_2 = distances_2.copy()
        for i in range(1, len(dx_2)):
            dx_2[i] = dx_2[i] - dx_2[i-1]

        self.lower_axis = dx_2

        dx_2[0] /= 2
        dx_2[-1] /= 2
        intMatrix_2 = matrix_2 @ dx_2
        
        coeffLiftMatrix = intMatrix_2 - intMatrix_1

        
        turners = np.cos(self.x_axis * np.pi / 180)
        coeffLiftMatrix = coeffLiftMatrix * turners
        
        return coeffLiftMatrix
    
    def plot_CL(self):
        plt.plot(self.x_axis, self.CL, label="CL")
        plt.plot(self.angles_for_actual_experiment, self.lift_cofficient_for_actual_experiment, label="Actual CL")
        plt.xlabel("Angle of Attack (deg)")
        plt.ylabel("Coefficient of Lift")
        plt.title("Coefficient of Lift vs Angle of Attack")
        plt.legend()

if __name__ == "__main__":
    df = pd.read_csv("Lab_data.csv", encoding="latin1")
    pressure_instance = pressureAnalysis(df)
    pressure_instance.plot_CL()
    plt.show()
