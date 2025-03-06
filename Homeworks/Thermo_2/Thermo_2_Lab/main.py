import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
'''

'''

def best_fit_curve(x, y, degree):

    coeffs = np.polyfit(x, y, degree)
    
    poly_eq = np.poly1d(coeffs)
    
    x_smooth = np.linspace(min(x), max(x), 100)
    y_smooth = poly_eq(x_smooth)  

    return x_smooth, y_smooth

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([2, 3, 5, 8, 12, 18, 26, 36, 50])  


data = pd.read_csv('data.csv')
headers = [
    "Time", "SPEED", "THRUST", "THROTTLE", "AIR_VOL_FLOW", "AIR_MASS_FLOW", "AIR_DENSITY",
    "FUEL_VOL_FLOW", "FUEL_MASS_FLOW", "FUEL_DENSITY", "AFR", "BAROMETRIC", "ORIFICE_DELTA",
    "PS_COMP_IN", "PS_DIFF_IN", "PS_DIFF_OUT", "PT_DIFF_OUT", "PT_NOZZLE_IN", "PT_EXHAUST",
    "T_AIR", "T_FUEL", "T_COMP_IN", "T_DIFF_IN", "T_DIFF_OUT", "T_NOZZLE_IN", "T_TURBINE_OUT", "T_EXHAUST"
]

data.columns = headers
data = data.dropna()

Throttle = data['THROTTLE'].to_numpy()
print(len(Throttle))
Thrust = data['THRUST'].to_numpy()

Fuel_mass_flow = data['FUEL_MASS_FLOW'].to_numpy()*0.001

specific_fuel_consumption = Fuel_mass_flow / Thrust 

throttle_smooth, thrust_smooth = best_fit_curve(Throttle, Thrust, 1)

_, fuel_mass_flow_smooth = best_fit_curve(Throttle, Fuel_mass_flow, 2)
_, sfc_smooth = best_fit_curve(Throttle, specific_fuel_consumption, 2)
_, thrust_smooth = best_fit_curve(Throttle, Thrust, 2)

min_fuel_consumption = np.min(sfc_smooth)
print(f'minimtm specific fuel consumption: {min_fuel_consumption}')

min_fuel_consumption_index = np.argmin(sfc_smooth)
min_fuel_consumption_throttle = throttle_smooth[min_fuel_consumption_index]
print(f'Throttle at minimum specific fuel consumption: {min_fuel_consumption_throttle}')
min_Thrust = thrust_smooth[min_fuel_consumption_index]
print(f'Thrust at minimum specific fuel consumption: {min_Thrust} N' )

fig, axes = plt.subplots(1, 3, figsize=(16, 4))  


axes[0].scatter(Throttle, Thrust, color='gray', s=10, label="Data")
axes[0].plot(throttle_smooth, thrust_smooth, color='red', label='Best-fit curve')
axes[0].set_xlabel(r'Throttle [$\%$]')
axes[0].set_ylabel(r'Thrust [N]')
axes[0].set_title('Throttle vs Thrust, (a)')
axes[0].grid()
axes[0].legend()

axes[1].scatter(Throttle, Fuel_mass_flow, color='gray', s=10, label="Data")
axes[1].plot(throttle_smooth, fuel_mass_flow_smooth, color='red', label='Best-fit curve')
axes[1].set_xlabel(r'Throttle [$\%$]')
axes[1].set_ylabel(r'Fuel mass flow [$\mathrm{kg/s}$]')
axes[1].set_title('Throttle vs Fuel Mass Flow, (b)')
axes[1].grid()
axes[1].legend()

axes[2].scatter(Throttle, specific_fuel_consumption, color='gray', s=10, label="Data")
axes[2].plot(throttle_smooth, sfc_smooth, color='red', label='Best-fit curve')
axes[2].set_xlabel(r'Throttle [$\%$]')
axes[2].set_ylabel(r'Specific fuel consumption [$\mathrm{kg/Ns}$]')
axes[2].set_title('Throttle vs Specific Fuel Consumption, (c)')
axes[2].grid()
axes[2].legend()

plt.tight_layout()
plt.show()

