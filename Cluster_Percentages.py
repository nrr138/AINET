import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

time = 60  # Define time interval for resampling

# Load and preprocess data
df = pd.read_feather("Data/Residential.feather")
df.set_index('index', inplace=True)
df = df.drop(df.index[:7200])  # Remove the first 7200 rows to start on "Monday"
columns = [col for col in df.columns if col.endswith('txavgbitrate') or col.endswith('rxavgbitrate')]
dfh = df[columns]  # Keep only columns containing 'txavgbitrate' & 'rxavgbitrate'
dfh = dfh.resample("" + str(time) + "Min").mean()  # Resample data
dfh = dfh.dropna()  # Drop NaN values
dfh.reset_index(inplace=True)
dfh = dfh.drop('index', axis=1)

# Pair rx and tx for every port
columns = dfh.columns
pairs = []
for col1 in columns:
    for col2 in columns:
        if col1 != col2 and col1[:16] == col2[:16]:
            if (col1, col2) not in pairs and (col2, col1) not in pairs:
                pairs.append((col1, col2))

# Calculate percentage data
percentage_data = {}
for rx, tx in pairs:
    total = dfh[rx] + dfh[tx]
    percentage_data[f'{tx}_percentage'] = dfh[tx] / total

dfh = pd.DataFrame(percentage_data)

df_t = pd.DataFrame()
ports = len(dfh.columns)

# Summarize data by time of day
for i in dfh:
    save_days_t = [0] * (24 * 7)
    for ind in range(len(dfh[i])):
        save_days_t[ind % (24 * 7)] = save_days_t[ind % (24 * 7)] + dfh[i][ind]

    df_t[i] = [x / ports for x in save_days_t]

# Weekday data
df_days = pd.DataFrame()

for i in df_t:
    for d in range(5):
        for h in range(3):
            df_days[i + str(h + 1) + '-' + str(d + 1)] = [df_t[i][j + 8 * h + 24 * d] for j in range(8)]

# Weekend data
df_weekend = pd.DataFrame()

for i in df_t:
    for d in range(2):
        for h in range(3):
            df_weekend[i + str(h + 1) + '-' + str(d + 1)] = [df_t[i][j + 120 + 8 * h + 24 * d] for j in range(8)]

# Prepare data for clustering
Xm = df_days.T.values

# Handle NaN values
for i in range(len(Xm)):
    for ii in range(len(Xm[0])):
        if math.isnan(Xm[i][ii]):
            Xm[i][ii] = 0

n_clusters = 100  # Number of clusters

# Initialize and fit KMeans model
kmeans = KMeans(n_clusters=n_clusters)
labelsm = kmeans.fit_predict(Xm)  # Predict cluster labels

percentages = [[0] * 3 for i in range(100)]
tot_save = [0] * 100

# Plot the clusters
plt.figure(figsize=(100, 100))
for i in range(int(len(Xm))):
    for k in range(100):
        if labelsm[i] == k:
            plt.subplot(10, 10, k + 1)
            plt.plot(range(8), Xm[i][:8], '.-')
            percentages[k][i % 3] += 1
            psum = sum(percentages[k])
            tot_save[k] = psum
            plt.title('Label ' + str(k) + f'\n{percentages[k][0] / psum:.2%}  {percentages[k][1] / psum:.2%}  {percentages[k][2] / psum:.2%}' + 'Total: ' + str(psum))
# plt.savefig('LabelsO_100.png')

sorted_list = sorted(enumerate(tot_save), key=lambda x: x[1], reverse=True)
sorted_indices = [x[0] for x in sorted_list]

plt.figure(figsize=(100, 100))
for i in range(int(len(Xm))):
    for k in range(len(sorted_indices)):
        if labelsm[i] == sorted_indices[k]:
            plt.subplot(10, 10, k + 1)
            plt.plot(range(8), Xm[i][:8], '.-')
            psum = sum(percentages[sorted_indices[k]])
            plt.title('Label ' + str(sorted_indices[k]) + f'\n{percentages[sorted_indices[k]][0] / psum:.2%}  {percentages[sorted_indices[k]][1] / psum:.2%}  {percentages[sorted_indices[k]][2] / psum:.2%}' + 'Total: ' + str(psum))
plt.savefig('Percentage.png')