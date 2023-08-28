import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Load and preprocess the data
df = pd.read_feather("Data/Residential.feather")
df.set_index('index', inplace=True)
df = df.drop(df.index[:7200])  # Remove the first 7200 rows to start on "Monday"
df = df.filter(regex='.*txavgbitrate.*')  # Keep only columns containing 'txavgbitrate'
df = df.resample("1W").mean()  # Resample the data to weekly intervals
df = df.fillna(0)  # Fill missing values with zeros

# Choose wavelet function
wavelet = 'db4'  # Daubechies 4

# Determine the grid layout for plots
num_ports = len(df.columns)
num_rows = 3
num_cols = 4
num_plots = num_rows * num_cols
num_grids = (num_ports + num_plots - 1) // num_plots

for grid in range(num_grids):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20))
    
    for i, col in enumerate(df.columns[grid*num_plots:(grid+1)*num_plots]):
        port_data = df[col]  # Select data for the current port
        coeffs = pywt.wavedec(port_data, wavelet)  # Perform wavelet decomposition
        
        # Calculate the power spectral density (PSD) of each level
        psd = []
        for coeff in coeffs:
            power = np.abs(coeff) ** 2
            psd.append(power)
        
        # Plot the PSD for the current port
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        for j, power in enumerate(psd):
            freqs = np.linspace(0, 1, len(power))
            ax.plot(freqs, power, label=f'Level {j+1}')
        ax.set_xlabel('Normalized Frequency')
        ax.set_ylabel('Power')
        ax.legend(prop={"size": 7}, loc="upper right")
        
    plt.show()
