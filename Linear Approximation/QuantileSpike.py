import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import QuantileRegressor
from joblib import Parallel, delayed

# Load and preprocess data
df = pd.read_feather('Data/Residential.feather')
df.set_index('index', inplace=True)
columns = [col for col in df.columns if col.endswith('txavgbitrate')]
df = df[columns]  # Keep only columns containing 'txavgbitrate'

# Initialize lists to store parameter values
mp1, kp1, dp1, dh1, mp2, kp2, dp2, dh2, mp3, kp3, dp3, dh3, mp4, kp4, dp4, dh4 = [[] for _ in range(16)]

window_size = 360  # Window size
max_trials = 500  # Maximum trials when fitting

# Fit subset of data using Quantile Regression
def fit_subset(y_data):
    if np.isnan(y_data).any():
        return None, None
    x_data = np.array(range(window_size)).reshape(-1, 1)
    quantile = QuantileRegressor(quantile=0.5, solver='highs-ds')
    quantile.fit(x_data, y_data)
    
    # Calculate residuals
    residual_data = y_data - quantile.predict(x_data)
    
    # Initialize lists to store positive and negative "spike" values
    positive_heights = []
    negative_heights = []
    for i in residual_data:
        if i > 0:
            positive_heights.append(i)
        elif i < 0:
            negative_heights.append(i)
            
    fig, ax = plt.subplots()

    # original data
    ax.plot(y_data, color='tab:blue')

    # Linear approximation
    linear_approximation = quantile.predict(x_data)
    ax.plot(linear_approximation, color='tab:green')

    # Positive spikes
    for i in range(len(positive_heights)):
        y_val = positive_heights[i] + linear_approximation[np.where(residual_data == positive_heights[i])[0][0]]
        ax.plot(np.where(residual_data == positive_heights[i])[0][0], y_val, 'r^')

    # Negative spikes
    for i in range(len(negative_heights)):
        y_val = negative_heights[i] + linear_approximation[np.flatnonzero(residual_data == negative_heights[i])[0]]
        ax.plot(np.flatnonzero(residual_data == negative_heights[i]), y_val, 'bv')

    plt.show()
    
    return quantile.intercept_, round(quantile.coef_[0], 2), residual_data

ii = 1
for col in df.columns:
    print(len(df.columns)-ii)
    ii += 1
    y = df[col].values
    results = Parallel(n_jobs=-1)(delayed(fit_subset)(y[i:i+window_size]) for i in range(0, len(y)-window_size, window_size))
    for itr, result in enumerate(results):
        if result[0] is not None:
            
            dirac_heights = np.array(result[2])
            dirac_points = np.arange(len(dirac_heights))
            
            dirp = dirac_points
            dirh = dirac_heights
            
            """ max_val = np.max(dirac_heights) # find the maximum height value
            min_val = abs(np.min(dirac_heights)) # find the minimum height value
            
            val = max([max_val, min_val])
            threshold = val * 1   # calculate the 25% threshold

            # use boolean indexing to filter the lists
            dirp = dirac_points[dirac_heights >= threshold]
            dirh = dirac_heights[dirac_heights >= threshold] """
            
            if itr % 4 == 0:
                mp1.append(result[0])
                kp1.append(result[1])
                dp1.append(dirp)
                dh1.append(dirh)
            elif itr % 4 == 1:
                mp2.append(result[0])
                kp2.append(result[1])
                dp2.append(dirp)
                dh2.append(dirh)
            elif itr % 4 == 2:
                mp3.append(result[0])
                kp3.append(result[1])
                dp3.append(dirp)
                dh3.append(dirh)
            elif itr % 4 == 3:
                mp4.append(result[0])
                kp4.append(result[1])
                dp4.append(dirp)
                dh4.append(dirh)

pm = [mp1, kp1, dp1, dh1, mp2, kp2, dp2, dh2, mp3, kp3, dp3, dh3, mp4, kp4, dp4, dh4]
min_length = min(len(x) for x in pm)
for i in range(len(pm)):
    pm[i] = pm[i][:min_length]

data = pd.DataFrame({
    'M1': pm[0],
    'K1': pm[1],
    'DP1': pm[2],
    'DH1': pm[3],
    'M2': pm[4],
    'K2': pm[5],
    'DP2': pm[6],
    'DH2': pm[7],
    'M3': pm[8],
    'K3': pm[9],
    'DP3': pm[10],
    'DH3': pm[11],
    'M4': pm[12],
    'K4': pm[13],
    'DP4': pm[14],
    'DH4': pm[15],
    })

data.to_feather('Parameters.feather')

kp1 = [x for x in kp1 if -1.2 <= x <= 1.2]
kp2 = [x for x in kp2 if -1.2 <= x <= 1.2]
kp3 = [x for x in kp3 if -1.2 <= x <= 1.2]
kp4 = [x for x in kp4 if -1.2 <= x <= 1.2]


"""plt.subplot(4,2,1)
plt.hist(mp1, bins=240)
plt.subplot(4,2,2)
plt.hist(kp1, bins=240)
plt.subplot(4,2,3)
plt.hist(mp2, bins=240)
plt.subplot(4,2,4)
plt.hist(kp2, bins=240)
plt.subplot(4,2,5)
plt.hist(mp3, bins=240)
plt.subplot(4,2,6)
plt.hist(kp3, bins=240)
plt.subplot(4,2,7)
plt.hist(mp4, bins=240)
plt.subplot(4,2,8)
plt.hist(kp4, bins=240)
plt.savefig('Parameters.png')
plt.savefig("Parameters.svg", format="svg")"""

"""plt.subplot(4,2,1)
plt.hist(mp1, bins=240)
plt.yscale('log')
plt.subplot(4,2,2)
plt.hist(kp1, bins=240)
plt.yscale('log')
plt.subplot(4,2,3)
plt.hist(mp2, bins=240)
plt.yscale('log')
plt.subplot(4,2,4)
plt.hist(kp2, bins=240)
plt.yscale('log')
plt.subplot(4,2,5)
plt.hist(mp3, bins=240)
plt.yscale('log')
plt.subplot(4,2,6)
plt.hist(kp3, bins=240)
plt.yscale('log')
plt.subplot(4,2,7)
plt.hist(mp4, bins=240)
plt.yscale('log')
plt.subplot(4,2,8)
plt.hist(kp4, bins=240)
plt.yscale('log')
plt.savefig('ParametersLog.png')
plt.savefig("ParametersLog.svg", format="svg")"""

"""plt.subplot(4,1,1)
plt.hist(dp1, bins=360)
plt.subplot(4,1,2)
plt.hist(dh1, bins=360)
plt.subplot(4,2,3)
plt.hist(dp2, bins=360)
plt.subplot(4,2,4)
plt.hist(dh2, bins=360)
plt.subplot(4,2,5)
plt.hist(dp3, bins=360)
plt.subplot(4,2,6)
plt.hist(dh3, bins=360)
plt.subplot(4,2,7)
plt.hist(dp4, bins=360)
plt.subplot(4,2,8)
plt.hist(dh4, bins=360)
plt.savefig('ParametersD.png')
plt.savefig("ParametersD.svg", format="svg")"""
