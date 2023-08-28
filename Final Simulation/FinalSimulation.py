import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

data = True
time = 8*60  # 8 Hour intervals
# Load data
if data == True:
    df = pd.read_feather("Data/OfficeAll.feather")
else:
    df = pd.read_feather("Data/ResidentialAll.feather")

# Process and clean data
df.set_index('index', inplace=True)
df = df.drop(df.index[:7200])
columnsr = [col for col in df.columns if col.endswith('txavgbitrate')]
dfhr = df[columnsr]
dfhr = dfhr.fillna(0)
dfh = dfhr.resample(""+str(time)+"Min").mean()
dfh.reset_index(inplace=True)
dfh = dfh.drop('index', axis=1)

# Define range sizes for grouping values
range_sizes = [1000] * 100 + [10000] * 90 + [100000] * 90 + [10**10]
dict = {}

# Group values based on ranges defined by range_sizes
for z in range(len(dfh.columns)):
    arr = dfh.iloc[:, z].values.tolist()
    start = 0
    for size in range_sizes:
        end = start + size
        values = []
        for i in range(len(arr)-1):
            if start <= arr[i] < end:
                values.append(arr[i+1])
        dict[start, end] = values
        start = end 

# Process data 
time = 10
time_set = int(60/time)
dfhr = dfhr.resample(""+str(time)+"Min").mean()
dfhr.reset_index(inplace=True)
dfhr = dfhr.drop('index', axis=1)

# Create a new DataFrame df_t
df_t = pd.DataFrame()
for i in dfhr:
    save_days_t = [0]*(time_set*24*7)
    for ind in range(len(dfhr[i])):
        save_days_t[ind%(time_set*24*7)] = save_days_t[ind%(time_set*24*7)] + dfhr[i][ind]
    df_t[i] = save_days_t

# Create a DataFrame df_days representing daily data
df_days = pd.DataFrame()
for i in df_t:
    for d in range(5):
        for h in range(3):
            df_days[i + str(h+1) +'-'+ str(d+1)] = [df_t[i][j + time_set*8*h+24*d] for j in range(time_set*8)]

# Create a DataFrame df_weekend representing weekend data
df_weekend = pd.DataFrame()
for i in df_t:
    for d in range(2):
        for h in range(3):
            df_weekend[i + str(h+1) +'-'+ str(d+1)] = [df_t[i][j + 120 + time_set*8*h+24*d] for j in range(time_set*8)]

# Extract weekdays and weekdays
Xm = df_days.iloc[:,:15].T.values
Xm_end = df_weekend.iloc[:,:6].T.values

# Get trend for time frames (00:01-08:00, 08:01-16:00, 16:01-24:00)
def get_trend(t, weekend):
    x = random.randint(0, len(Xm_end)-3) if weekend else random.randint(0, len(Xm)-3)
    if t == 0:
        xx = Xm[x]
    elif t == 1:
        xx = Xm[x+1]
    elif t == 2:
        xx = Xm[x+2]
    return xx

# Define a function to generate normal values from predefined ranges
def get_normal_value(val):

    # Find the appropriate range in 'dict'
    key_found = False
    for key in dict:
        if key[0] <= val < key[1] and dict[key] != []:
            key_found = True
            values = dict[key]
            break

    # If the value doesn't fit in any range, find the closest range and use its values to generate a random value
    if not key_found:
        closest_range = None
        closest_distance = float('inf')
        for key in dict:
            distance = min(abs(val - key[0]), abs(val - key[1]))
            if distance < closest_distance and len(dict[key]) > 0:
                closest_range = key
                closest_distance = distance
        if closest_range is not None:
            values = dict[closest_range]
        else:
            return None
        
    # Generate a new value based on the range mean and std
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

# Define a function to adjust an arrays average to a new target average
def change_list_average(array, new_average):
    current_average = np.mean(array)
    difference = new_average - current_average
    scaled_array = array + difference * (array / current_average)
    return scaled_array.tolist()

# Initialize the simulation arrays
sim = [arr[0]]
sim2 = []

# Perform simulation for 7 days
for day in range(7):
    for i in range(3):    
        val = get_normal_value(sim[-1])
        sim.append(val)

        # Choose a trend based on the day (weekend or not)
        if day%5 == 0 or day%6 == 0:
            trend = get_trend(i%3, True)
        else:
            trend = get_trend(i%3, False)
            
        sim2 = sim2 + change_list_average(trend, val)

# Plot the simulation results
plt.figure(2)
plt.plot(sim2)  # Simulated + time
plt.plot(dfhr.iloc[:, 7])  # Original data
plt.xlim([-10, len(sim2)+10])    
plt.legend(['Simulation', 'Original'])
plt.ylabel('Received bitrate')
plt.xlabel('Time ('+str(time)+'-minute)')
plt.show()

# Convert the simulated data to a DataFrame and save as feather file
simm2 = pd.DataFrame(sim2)
simm2.columns = ['Simulated']
simm2.to_feather('Simulering save/simulation.feather')
