import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parse .xvg files and extract numerical data
def parse_xvg(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith(("#", "@")):
                values = line.split()
                if len(values) >= 2:
                    try:
                        data.append([float(v) for v in values])
                    except ValueError:
                        continue
    return pd.DataFrame(data)

# Clean and extract numerical data from XVG files
def clean_xvg(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith(('#', '@', '/*', 'static')):
                try:
                    data.append([float(x) for x in line.split()])
                except ValueError:
                    continue
    return pd.DataFrame(data)

# File paths
msd_file = "msd.xvg"
density_file = "density.xvg"
gyrate_file = "gyrate.xvg"
sasa_file = "sasa_cleaned.xvg"
rmsd_file = "rmsd-dist.xvg"

# Parse data from the files
msd_data = parse_xvg(msd_file)
density_data = parse_xvg(density_file)
gyrate_data = parse_xvg(gyrate_file)
sasa_data = clean_xvg(sasa_file)
rmsd_data = np.loadtxt(rmsd_file, comments=["@", "#"])

# Assign column names
msd_data.columns = ["Time (ps)", "MSD (nm^2)"]
density_data.columns = ["Coordinate (nm)", "Density (kg/m^3)"]
gyrate_data.columns = ["Time (ps)", "Rg Total (nm)", "Rg X (nm)", "Rg Y (nm)", "Rg Z (nm)"]
sasa_data.columns = ["Time (ps)", "SASA (nm^2)"]

# Convert time units
msd_data["Time (ns)"] = msd_data["Time (ps)"] / 1000
gyrate_data["Time (ns)"] = gyrate_data["Time (ps)"] / 1000
sasa_data["Time (ns)"] = sasa_data["Time (ps)"] / 1000

# SASA Polynomial and Linear Fits
x_ns = sasa_data["Time (ns)"]
y = sasa_data["SASA (nm^2)"]
poly_coeffs = np.polyfit(x_ns, y, 16)
poly_eq = np.poly1d(poly_coeffs)
y_poly = poly_eq(x_ns)
linear_coeffs = np.polyfit(x_ns, y, 1)
linear_eq = np.poly1d(linear_coeffs)
y_linear = linear_eq(x_ns)

# Load RMSD Data
time_rmsd = rmsd_data[:, 0]
rmsd = rmsd_data[:, 1]

# Load Vesicle Radius Data
time_gyrate, radius, rg_x, rg_y, rg_z = np.loadtxt(gyrate_file, comments=["@", "#"], unpack=True)
time_gyrate_ns = time_gyrate / 1000

# Plot all graphs
plt.figure(figsize=(12, 15))

# Plot 1: Mean Squared Displacement (MSD) over Time
plt.subplot(3, 2, 1)
plt.plot(msd_data["Time (ns)"], msd_data["MSD (nm^2)"], label="MSD", color="blue")
plt.xlabel("Time (ns)")
plt.ylabel("MSD (nm^2)")
plt.title("Mean Squared Displacement")
plt.legend()

# Plot 2: Density Distribution per Coordinate
plt.subplot(3, 2, 2)
plt.plot(density_data["Coordinate (nm)"], density_data["Density (kg/m^3)"], label="Density", color="red")
plt.xlabel("Coordinate (nm)")
plt.ylabel("Density (kg/m^3)")
plt.title("Density per Coordinate")
plt.legend()

# Plot 3: Radius of Gyration Over Time
plt.subplot(3, 2, 3)
plt.plot(gyrate_data["Time (ns)"], gyrate_data["Rg Total (nm)"], label="Total Rg", color="green")
plt.plot(gyrate_data["Time (ns)"], gyrate_data["Rg X (nm)"], label="Rg X", linestyle="dashed", color="orange")
plt.plot(gyrate_data["Time (ns)"], gyrate_data["Rg Y (nm)"], label="Rg Y", linestyle="dotted", color="purple")
plt.plot(gyrate_data["Time (ns)"], gyrate_data["Rg Z (nm)"], label="Rg Z", linestyle="dashdot", color="brown")
plt.xlabel("Time (ns)")
plt.ylabel("Radius of Gyration (nm)")
plt.title("Radius of Gyration Over Time")
plt.legend()

# Plot 4: Solvent Accessible Surface Area (SASA) Over Time
plt.subplot(3, 2, 4)
plt.scatter(x_ns, y, label="SASA Data", color='c', s=10)
plt.plot(x_ns, y_poly, label="Polynomial Fit", color='b', linestyle='dashed')
plt.plot(x_ns, y_linear, label="Linear Fit", color='r', linestyle='dotted')
plt.xlabel("Time (ns)")
plt.ylabel("SASA (nm^2)")
plt.title("Solvent Accessible Surface Area Over Time")
plt.legend()
plt.grid(True)

# Plot 5: RMSD Distribution
plt.subplot(3, 2, 5)
plt.plot(time_rmsd, rmsd, linestyle="-", color="b")
plt.xlabel("RMS (nm)")
plt.ylabel("Counts")
plt.title("Counts per RMS")
plt.grid(True)

# Plot 6: Vesicle Radius Over Time
plt.subplot(3, 2, 6)
plt.plot(time_gyrate_ns, radius, linestyle='-', label="Radius of Gyration (Rg)", color='m')
plt.xlabel("Time (ns)")
plt.ylabel("Radius (nm)")
plt.title("Vesicle Radius Over Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
