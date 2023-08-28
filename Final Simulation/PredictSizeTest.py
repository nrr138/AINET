import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

data = False
time = 30  # 30-minute intervals

# Check data and load appropriate data
if data == True:
    df = pd.read_feather("Data/OfficeAll.feather")
else:
    df = pd.read_feather("Data/ResidentialAll.feather")

# Preprocces and clean data
df.set_index('index', inplace=True)
columnsr = [col for col in df.columns if col.endswith('txavgbitrate')]
dfhr = df[columnsr] # Create a DataFrame containing only the selected columns
dfhr = dfhr.resample(""+str(time)+"Min").mean()
dfhr = dfhr.fillna(0)
dfhr.reset_index(inplace=True)
dfhr = dfhr.drop('index', axis=1)

# Define different range sizes for grouping values
range_sizes = [1000] * 100 + [10000] * 90 + [100000] * 90 + [10**10]
dict = {}  # Initialize a dictionary to store value ranges

# Extract the values from the selected column
arr = dfhr['po100kw1__eth9__txavgbitrate'].values.tolist()

# Group values into different ranges and store in the dictionary
start = 0
for size in range_sizes:
    end = start + size
    values = []
    for i in range(len(arr)-1):
        if start <= arr[i] < end:
            values.append(arr[i+1])
    dict[start, end] = values
    start = end

# Get a normally distributed value from the dictionary
def get_normal_value(val):
    key_found = False

    # Find the key that corresponds to the range of the input value
    for key in dict:
        if key[0] <= val < key[1] and dict[key] != []:
            key_found = True
            values = dict[key]
            break

    if not key_found:
        closest_range = None
        closest_distance = float('inf')

        # Find the closest range with values if input value is not in any range
        for key in dict:
            distance = min(abs(val - key[0]), abs(val - key[1]))
            if distance < closest_distance and len(dict[key]) > 0:
                closest_range = key
                closest_distance = distance
        if closest_range is not None:
            values = dict[closest_range]
        else:
            return None

    if len(values) >= 2:
        mean = np.mean(values)
        std = np.std(values)
        value = float(random.normalvariate(mean, std))
        if value < 0:
            return 0.0
        else:
            return value
    elif len(values) == 1:
        return values[0]
    else:
        return None

# Simulate new data points using "get_normal_value" 
sim = [arr[0]]
for _ in range(len(arr)):
    sim.append(get_normal_value(sim[-1]))

# Plot the simulation and original data
plt.plot(sim)
plt.plot(arr)
plt.legend(['Simulation', 'Original'])
plt.ylabel('Received bitrate')
plt.xlabel('Time (30min)')
plt.xlim([0, 336])  # Limit x-axis range
plt.show()