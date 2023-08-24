# Clusters each day separately

import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering

time = 60 # Define time interval for resampling

# Load and preprocess data
df = pd.read_feather("Data/Residential.feather")
df.set_index('index', inplace=True)
df = df.drop(df.index[:7200])  # Remove the first 7200 rows to start on "Monday"
columns = [col for col in df.columns if col.endswith('txavgbitrate')]
dfh = df[columns]  # Keep only columns containing 'txavgbitrate'
dfh = dfh.resample(""+str(time)+"Min").mean()  # Resample data
dfh = dfh.dropna()  # Drop NaN values
dfh.reset_index(inplace=True)
dfh = dfh.drop('index', axis=1)

df_t = pd.DataFrame()
df_t1 = pd.DataFrame()

# Separate days
for i in dfh:
    save_days_t = [0]*(24*7)
    for ind in range(len(dfh[i])):
        save_days_t[ind%(24*7)] = save_days_t[ind%(24*7)]+ dfh[i][ind]

    df_t[i] = save_days_t

df_mon = pd.DataFrame()
df_tue = pd.DataFrame()
df_wed = pd.DataFrame()
df_thu = pd.DataFrame()
df_fri = pd.DataFrame()

# Separate time intervals
for i in df_t:
    df_mon[i + str(1)] = [df_t[i][j] for j in range(8)]
    df_mon[i + str(2)] = [df_t[i][j+8] for j in range(8)]
    df_mon[i + str(3)] = [df_t[i][j+16] for j in range(8)]

# Scale data by total sum
def ratio_scale(df):
    scaled_df = df.copy()
    column_totals = df.sum(axis=0)
    scaled_df = df.div(column_totals, axis=1)
    return scaled_df

Xm = ratio_scale(df_mon).T.values

# Handle NaN values
for i in range(len(Xm)):
    for ii in range(len(Xm[0])):
        if math.isnan(Xm[i][ii]):
            Xm[i][ii] = 0
            
# Initialize SSE list
sse = []

# Try clustering with different number of clusters from 1 to 10
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(Xm)
    sse.append(kmeans.inertia_)  # Inertia is SSE

# Plot SSE vs number of clusters
plt.figure()
plt.plot(range(1, 20), sse)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.title('Elbow Method')
plt.savefig('Elbow.png')

# Average silhouette method
silhouette_scores = []
for n_clusters in range(2, 100):
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(Xm)
    silhouette_scores.append(silhouette_score(Xm, cluster_labels))

plt.figure()
plt.plot(range(2, 100), silhouette_scores,'o-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.savefig('Silhuette.png')