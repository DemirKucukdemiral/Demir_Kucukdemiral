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
Thrust = data['THRUST'].to_numpy()

Fuel_mass_flow = data['FUEL_MASS_FLOW'].to_numpy()*0.001

specific_fuel_consumption = Fuel_mass_flow / Thrust 

throttle_smooth, thrust_smooth = best_fit_curve(Throttle, Thrust, 1)

plt.scatter(Throttle, Thrust, color='gray', s=1)
plt.plot(throttle_smooth, thrust_smooth, color='red', label='Best-fit curve')
plt.xlabel(r'Throttle [$\%$]')
plt.ylabel(r'Thrust [N]')
plt.grid()
plt.show()

_, fuel_mass_flow_smooth = best_fit_curve(Throttle, Fuel_mass_flow, 2)


plt.scatter(Throttle, Fuel_mass_flow, color='gray', s=1)
plt.plot(throttle_smooth, fuel_mass_flow_smooth, color='red', label='Best-fit curve')
plt.xlabel(r'Throttle [$\%$]')
plt.ylabel(r'Fuel mass flow [$\mathrm{kgs^{-1}}$]')
plt.grid()
plt.show()

_, sfc_smooth = best_fit_curve(Throttle, specific_fuel_consumption, 2)

plt.scatter(Throttle, specific_fuel_consumption, color='gray', s=1)
plt.plot(throttle_smooth, sfc_smooth, color='red', label='Best-fit curve')
plt.xlabel(r'Throttle [$\%$]')
plt.ylabel(r'Specific fuel consumption [$\mathrm{kgN^{-1}s^{-1}}$]')
plt.grid()
plt.show()

min_fuel_consumption = np.min(specific_fuel_consumption)
print(f'minimtm specific fuel consumption: {min_fuel_consumption}')

min_fuel_consumption_index = np.argmin(specific_fuel_consumption)
min_fuel_consumption_throttle = Throttle[min_fuel_consumption_index]
print(f'Throttle at minimum specific fuel consumption: {min_fuel_consumption_throttle}')
