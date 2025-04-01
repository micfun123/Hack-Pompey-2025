import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

def get_portsmouth_sigmas(x, stability_class='D'):
    """
    Returns sigma_y and sigma_z for Portsmouth with atmospheric stability adjustment.
    x: Downwind distance (m).
    stability_class: Pasquill-Gifford stability class (A-F, where A=very unstable, F=very stable).
    """
    stability_factors = {
        'A': (0.22, 0.2), 'B': (0.16, 0.12), 'C': (0.11, 0.08),
        'D': (0.08, 0.06), 'E': (0.06, 0.03), 'F': (0.04, 0.016)
    }
    factor_y, factor_z = stability_factors.get(stability_class, (0.08, 0.06))
    sigma_y = factor_y * x / np.sqrt(1 + 0.0001 * x)
    sigma_z = factor_z * x / np.sqrt(1 + 0.0015 * x)
    return sigma_y, sigma_z

def gaussian_plume(x_grid, y_grid, Q, wind_speed, wind_dir, sigma_y, sigma_z, H=76, deposition_rate=0.01, decay_rate=0.001):
    """
    Computes the concentration of particles at grid points with deposition and decay.
    """
    if wind_speed <= 0:
        return np.zeros_like(x_grid)
    
    wind_dir_rad = np.radians(wind_dir)
    x_rot = x_grid * np.cos(wind_dir_rad) + y_grid * np.sin(wind_dir_rad)
    y_rot = -x_grid * np.sin(wind_dir_rad) + y_grid * np.cos(wind_dir_rad)
    
    C = (Q / (2 * np.pi * wind_speed * sigma_y * sigma_z)) * \
        np.exp(-0.5 * (y_rot / sigma_y) ** 2) * \
        np.exp(-0.5 * (((x_rot - 0) / sigma_z) ** 2)) * \
        np.exp(-decay_rate * x_rot) * np.exp(-deposition_rate * x_rot)
    
    return C / np.max(C)  # Normalize for visualization

def gaussian_plume_to_transparent_image(gaussian_plume):
    plt.figure(figsize=(6, 6), dpi=300, facecolor='none')
    plt.imshow(gaussian_plume, cmap='hot', interpolation='nearest', alpha=0.75, origin='lower')
    plt.axis('off')
    plt.savefig("gaussian_plume.png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

data = json.load(open("output.json"))

def makedfromjson(jsondata):
    rows = []
    for date, values in jsondata.items():
        rows.append((date, float(values["Wind Speed"]), float(values["Wind Dir"]),
                     float(values["PM1"]), float(values["PM10"]), float(values["PM25"])) )
    df = pd.DataFrame(rows, columns=["date", "Wind Speed", "Wind Dir", "PM1", "PM10", "PM25"])
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df

df = makedfromjson(data)
first_entry = df.iloc[5]

x_vals = np.linspace(-500, 500, 100)
y_vals = np.linspace(-500, 500, 100)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)

sigma_y, sigma_z = get_portsmouth_sigmas(1000, stability_class='D')
c = gaussian_plume(x_grid, y_grid, first_entry["PM1"], first_entry["Wind Speed"], first_entry["Wind Dir"],
                    sigma_y, sigma_z)

# Make black areas fully transparent
plt.figure(figsize=(6, 6), dpi=300, facecolor='none')
plt.imshow(c, cmap='hot', interpolation='nearest', alpha=0.75, origin='lower', extent=[-500, 500, -500, 500])
plt.axis('off')
plt.savefig("gaussian_plume.png", bbox_inches='tight', pad_inches=0, transparent=True)
plt.close()
