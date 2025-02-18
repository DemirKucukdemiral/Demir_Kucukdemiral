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
        print(f"Mean velocity: {self.velocity_mean.round(2)} m/s")

        self.x_axis = np.linspace(-6, 20, 14)
        
        self.pressure_data = data.iloc[4:, 12:32].reset_index(drop=True)
        self.pressure_data = self.pressure_data.apply(pd.to_numeric, errors='coerce')
        self.pressure_data.to_csv("pressure_data.csv", index=False)

        # Correcting specific columns (averaging known faulty tappings)
        self.pressure_data.iloc[:, 7]  = (self.pressure_data.iloc[:, 9] + self.pressure_data.iloc[:, 5]) / 2
        self.pressure_data.iloc[:, 8]  = (self.pressure_data.iloc[:, 10] + self.pressure_data.iloc[:, 6]) / 2
        self.pressure_data.iloc[:, 16] = (self.pressure_data.iloc[:, 14] + self.pressure_data.iloc[:, 18]) / 2

        self.chord = 150  # mm

        self.upper_surface_pressures = self.pressure_data.iloc[:, 1::2]
        self.lower_surface_pressures = self.pressure_data.iloc[:, ::2]

        self.upper_surface_pressures_matrix = self.upper_surface_pressures.to_numpy()
        self.lower_surface_pressures_matrix = self.lower_surface_pressures.to_numpy()
    
        self.lowerDistances = np.array([0.76, 3.81, 11.43, 19.05, 38, 62, 80.77, 101.35, 121.92, 137.16])
        self.upperDistances = np.array([1.52, 7.62, 12.24, 22.86, 41.15, 59.44, 77.73, 96.02, 114.30, 129.54])
        
        # Compute Cp
        self.Cp_upper = self.coefficientPressure(self.upper_surface_pressures_matrix) * 1000
        self.Cp_lower = self.coefficientPressure(self.lower_surface_pressures_matrix) * 1000

        # Integrate to find CL
        self.CL = self.trapezoidal_rule(self.Cp_upper, self.upperDistances, self.Cp_lower, self.lowerDistances)

        # Reynolds number
        self.reynolds_number = self.air_density * self.velocity_mean * self.chord / (1.81e-5 * 1000)
        print(f"Reynolds number: {self.reynolds_number.round(2)}")
        
        # Load reference data for comparison
        data_naca0012 = np.loadtxt("naca0012_experiment_re_200000.txt")
        self.angles_for_actual_experiment = data_naca0012[:, 0] 
        self.lift_cofficient_for_actual_experiment = data_naca0012[:, 1]

        # === NEW: Plot multiple Cp graphs as subplots ===
        self.plot_all_cp_subplots()

    def coefficientPressure(self, pressure_to_plot):
        ones_matrix = np.ones_like(pressure_to_plot)
        # Debug: print(pressure_to_plot)  # If you want to see raw values
        pressure_coefficient_matrix = (pressure_to_plot - self.pressure_atm * ones_matrix) \
                                      / (0.5 * self.air_density * self.velocity_mean**2)
        return pressure_coefficient_matrix

    def trapezoidal_rule(self, matrix_1, distances_1, matrix_2, distances_2):
        # Convert distances to fraction of chord
        distances_1 = np.array(distances_1) / self.chord
        dx_1 = distances_1.copy()
        for i in range(1, len(dx_1)):
            dx_1[i] = dx_1[i] - dx_1[i-1]

        self.upper_axis = dx_1
        dx_1[0]  /= 2
        dx_1[-1] /= 2
        intMatrix_1 = matrix_1 @ dx_1
        
        distances_2 = np.array(distances_2) / self.chord
        dx_2 = distances_2.copy()
        for i in range(1, len(dx_2)):
            dx_2[i] = dx_2[i] - dx_2[i-1]

        self.lower_axis = dx_2
        dx_2[0]  /= 2
        dx_2[-1] /= 2
        intMatrix_2 = matrix_2 @ dx_2
        
        coeffLiftMatrix = intMatrix_2 - intMatrix_1

        # Multiply by cos(AoA) factor if needed
        turners = np.cos(self.x_axis * np.pi / 180)
        coeffLiftMatrix = coeffLiftMatrix * turners
        
        return coeffLiftMatrix

    def plot_CL(self):
        """
        Plots the coefficient of lift vs. angle of attack,
        comparing the experimental results with known data.
        """
        plt.figure()
        plt.plot(self.x_axis, self.CL, label="CL (Experiment)")
        plt.plot(self.angles_for_actual_experiment, self.lift_cofficient_for_actual_experiment, 
                 label="CL (Reference)")
        plt.xlabel("Angle of Attack (deg)")
        plt.ylabel("Coefficient of Lift")
        plt.title("Coefficient of Lift vs Angle of Attack")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_all_cp_subplots(self):
       
        angles_of_attack = [f"{i}°" for i in self.x_axis]  
       

        # Subplot labels (a), (b), (c), ...
        subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)",
                          "(f)", "(g)", "(h)", "(i)", "(j)",
                          "(k)", "(l)", "(m)", "(n)", "(o)"]
        # You can extend or trim this list as needed.

        num_plots = len(self.Cp_upper)  # total number of Cp sets
        cols = 5
        rows = math.ceil(num_plots / cols)

        plt.figure(figsize=(15, 6))  # Adjust figure size as needed

        for i in range(num_plots):
            # If we run out of labels or angles, handle it gracefully
            label_text = subplot_labels[i] if i < len(subplot_labels) else ""
            angle_text = angles_of_attack[i] if i < len(angles_of_attack) else ""

            plt.subplot(rows, cols, i + 1)
            plt.plot(self.upperDistances / self.chord, self.Cp_upper[i], label="Upper Surface")
            plt.plot(self.lowerDistances / self.chord, self.Cp_lower[i], label="Lower Surface")
            plt.title(f"{label_text} AoA = {angle_text}")
            plt.xlabel("x/c")
            plt.ylabel("Cp")
            plt.grid(True)
            if i == 0:  # put legend on the first subplot, or change as you prefer
                plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv("Lab_data.csv", encoding="latin1")
    pressure_instance = pressureAnalysis(df)
    # Plot all Cp subplots (already called at the end of __init__)
    # Plot CL
    pressure_instance.plot_CL()