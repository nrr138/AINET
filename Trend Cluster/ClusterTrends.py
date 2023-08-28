import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

# Set time interval for resampling
time = 60  # 8 Hour intervals

# Load data and preprocess
df = pd.read_feather("Data/ResidentialAll.feather")
df.set_index('index', inplace=True)
df = df.drop(df.index[:7200])
columns = [col for col in df.columns if col.endswith('txavgbitrate')]
dfh = df[columns]
dfh = dfh.resample(""+str(time)+"Min").mean()  # Take mean of "time" minute intervals
dfh = dfh.fillna(0)
dfh.reset_index(inplace=True)  # Reset index
dfh = dfh.drop('index', axis=1)

# Create a DataFrame to save days
df_t = pd.DataFrame()
for i in dfh:
    save_days_t = [0]*(24*7)
    for ind in range(len(dfh[i])):
        save_days_t[ind%(24*7)] = save_days_t[ind%(24*7)] + dfh[i][ind]
    df_t[i] = save_days_t

# Create a DataFrame for weekdays
df_days = pd.DataFrame()
for i in df_t:
    for d in range(5):
        for h in range(3):
            df_days[i + str(h+1) +'-'+ str(d+1)] = [df_t[i][j + 8*h+24*d] for j in range(8)]

# Create a DataFrame for weekends
df_weekend = pd.DataFrame()
for i in df_t:
    for d in range(2):
        for h in range(3):
            df_weekend[i + str(h+1) +'-'+ str(d+1)] = [df_t[i][j + 120 + 8*h+24*d] for j in range(8)]

# Ratio scale Dataframe
def ratio_scale(df):
    scaled_df = df.copy()
    column_totals = df.sum(axis=0)
    scaled_df = df.div(column_totals, axis=1)
    return scaled_df

# Define a clustering function
def cluster_func(Xm, save_name):

    # Handle missing values by setting them to 0
    for i in range(len(Xm)):
        for ii in range(len(Xm[0])):
            if math.isnan(Xm[i][ii]):
                Xm[i][ii] = 0
    n_clusters = 100 # 100 clusters

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(Xm)
    labelsm = kmeans.fit_predict(Xm)

    # Calculate percentages and total counts for each cluster
    percentages = [[0]*3 for i in range(100)]
    tot_save = [0]*100
    for i in range(int(len(Xm))):
        for k in range(100):
            if labelsm[i] == k:
                percentages[k][i % 3] += 1
                psum = sum(percentages[k])
                tot_save[k] = psum

    # Sort clusters based on total counts
    sorted_list = sorted(enumerate(tot_save), key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in sorted_list]

    # Plot individual clusters and their percentages
    plt.figure(figsize=(100, 100))
    for i in range(int(len(Xm))):
        for k in range(len(sorted_indices)):
            if labelsm[i] == sorted_indices[k]:
                plt.subplot(10, 10, k+1)
                plt.plot(range(8), Xm[i][:8], '.-')
                psum = sum(percentages[sorted_indices[k]])
                plt.title('Label ' + str(sorted_indices[k]) + f'\n{percentages[sorted_indices[k]][0] / psum:.2%}  {percentages[sorted_indices[k]][1] / psum:.2%}  {percentages[sorted_indices[k]][2] / psum:.2%}' + 'Total: ' + str(psum))
                plt.ylim([0, 0.7])
    plt.savefig(save_name)

    # Calculate average values for different time segments within each cluster
    div = [0]*100
    ave = [[0]*8 for _ in range(100)]
    label_ave = [-1]*100
    for i in range(int(len(Xm))):
        for k in range(len(sorted_indices)):
            if labelsm[i] == sorted_indices[k]:
                l = Xm[i][:8]
                ave[labelsm[i]] = [ave[labelsm[i]][z] + l[z] for z in range(8)]
                div[labelsm[i]] += 1

                psum = sum(percentages[sorted_indices[k]])
                if psum > 9:
                    if (percentages[sorted_indices[k]][0] / psum) > 0.6:
                        label_ave[labelsm[i]] = 0
                    elif (percentages[sorted_indices[k]][1] / psum) > 0.6:
                        label_ave[labelsm[i]] = 1
                    elif (percentages[sorted_indices[k]][2] / psum) > 0.6:
                        label_ave[labelsm[i]] = 2
                    else:
                        label_ave[labelsm[i]] = 3

    # Plot average values for different time segments within each cluster
    plt.figure(figsize=(100, 100))
    for i in range(int(len(Xm))):
        for q in range(len(sorted_indices)):
            if labelsm[i] == sorted_indices[q]:
                plot = [ave[labelsm[i]][x]/div[labelsm[i]] for x in range(8)]
                plt.subplot(10, 10, q+1)
                plt.plot(range(8), plot, '.-')
                psum = sum(percentages[sorted_indices[q]])
                plt.ylim([0, 0.7])
                plt.title('Label ' + str(sorted_indices[q]) + f'\n{percentages[sorted_indices[q]][0] / psum:.2%}  {percentages[sorted_indices[q]][1] / psum:.2%}  {percentages[sorted_indices[q]][2] / psum:.2%}' + 'Total: ' + str(psum))
    plt.savefig(save_name+"_ave")

    # Plot average values for morning, day, night, and uncertain segments within each cluster
    plt.figure(figsize=(10, 10))
    plt_titles = ["Morning (0-8) - Total labels: " + str(label_ave.count(0)),
                  "Middle day (8-16) - Total labels: " + str(label_ave.count(1)),
                  "End of day (16-24) - Total labels: " + str(label_ave.count(2)),
                  "Uncertian - Total labels: " + str(label_ave.count(3))]
    for i in range(4):
        plt.subplot(2, 2, i+1)
        for q in range(100):
            if label_ave[q] == i:
                plot = [ave[q][x]/div[q] for x in range(8)]
                plt.plot(range(8), plot, '.-')
                plt.ylim([0, 0.7])
                plt.title(plt_titles[i])
    plt.savefig(save_name+"_average")

# Run the cluster code
Xm = ratio_scale(df_weekend).T.values
cluster_func(Xm, "Label_100_weekend")
Xm = ratio_scale(df_days).T.values
cluster_func(Xm, "Label_100_weekdays")
