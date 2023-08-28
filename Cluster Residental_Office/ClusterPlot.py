import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering

# Preprocess data 
def clean_feather(df):
    time = 480  
    df.set_index('index', inplace=True)
    df = df.drop(df.index[:7200])
    columns = [col for col in df.columns if col.endswith('txavgbitrate')]
    dfh = df[columns]
    dfh = dfh.resample("" + str(time) + "Min").mean()  # Resample data at 'time' minute intervals
    column_sums = dfh.sum(axis=0)
    for i in [col for col in column_sums.index if column_sums[col] < 100000]:
        dfh = dfh[dfh.columns.drop(i)]
    dfh = dfh.dropna()
    return dfh

# Load, clean, and concatenate multiple dataframes
df1 = clean_feather(pd.read_feather("ifcounts_be25d1.feather"))
df2 = clean_feather(pd.read_feather("ifcounts_sh1d1.feather"))
df3 = clean_feather(pd.read_feather("ifcounts_ca3d1.feather"))
df4 = clean_feather(pd.read_feather("ifcounts_po9d1.feather"))
df5 = clean_feather(pd.read_feather("ifcounts_bs3d1.feather"))
df6 = clean_feather(pd.read_feather("ifcounts_br6d1.feather"))
df7 = clean_feather(pd.read_feather("ifcounts_br1d1.feather"))
df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1)

# Reset index and drop unnecessary column
df.reset_index(inplace=True)
df = df.drop('index', axis=1)

# Accumulate data for each day of the week
df_t = pd.DataFrame()
for i in df:
    save_days_t = [0] * 21
    for ind in range(len(df[i])):
        save_days_t[ind % 21] = save_days_t[ind % 21] + df[i][ind]
    df_t[i] = save_days_t

# Split the data for each day of the week
df_mon = pd.DataFrame()
df_tue = pd.DataFrame()
df_wed = pd.DataFrame()
df_thu = pd.DataFrame()
df_fri = pd.DataFrame()
for i in df_t:
    df_mon[i] = [df_t[i][j] for j in range(3)]
    df_tue[i] = [df_t[i][j + 3] for j in range(3)]
    df_wed[i] = [df_t[i][j + 6] for j in range(3)]
    df_thu[i] = [df_t[i][j + 9] for j in range(3)]
    df_fri[i] = [df_t[i][j + 12] for j in range(3)]

# Scale data by ratio
def ratio_scale(df):
    scaled_df = df.copy()
    column_totals = df.sum(axis=0)
    scaled_df = df.div(column_totals, axis=1)
    return scaled_df

# Scale data for each day of the week
dfm = ratio_scale(df_mon)
dftu = ratio_scale(df_tue)
dfw = ratio_scale(df_wed)
dfth = ratio_scale(df_thu)
dff = ratio_scale(df_fri)

# Create arrays for clustering
Xm = dfm.T.values
Xtu = dftu.T.values
Xw = dfw.T.values
Xth = dfth.T.values
Xf = dff.T.values

# Initialize and fit clustering models
kmeansm = SpectralClustering(n_clusters=2)
kmeanstu = SpectralClustering(n_clusters=2)
kmeansw = SpectralClustering(n_clusters=2)
kmeansth = SpectralClustering(n_clusters=2)
kmeansf = SpectralClustering(n_clusters=2)
labelsm = kmeansw.fit_predict(Xm)
labelstu = kmeansw.fit_predict(Xtu)
labelsw = kmeansw.fit_predict(Xw)
labelsth = kmeansw.fit_predict(Xth)
labelsf = kmeansw.fit_predict(Xf)

# Create scatter plots for each day of the week
plt.subplot(2, 3, 1)
plt.scatter(Xm[:, 0], Xm[:, 1], c=labelsm)
plt.title('Monday')

plt.subplot(2, 3, 2)
plt.scatter(Xtu[:, 0], Xtu[:, 1], c=labelstu)
plt.title("Tuesday")

plt.subplot(2, 3, 3)
plt.scatter(Xw[:, 0], Xw[:, 1], c=labelsw)
plt.title("Wednesday")

plt.subplot(2, 3, 4)
plt.scatter(Xth[:, 0], Xth[:, 1], c=labelsth)
plt.title("Thursday")

plt.subplot(2, 3, 5)
plt.scatter(Xf[:, 0], Xf[:, 1], c=labelsf)
plt.title("Friday")
plt.show()

# Determine label meanings
def label(label_0):
    if label_0 == 1:
        lab = ['Residential', 'Office']
    else:
        lab = ['Office', 'Residential']
    return lab

# Determine labels for each day of the week
labm = label(labelsm[0])
labtu = label(labelstu[0])
labw = label(labelsw[0])
labth = label(labelsth[0])
labf = label(labelsf[0])

# Determine whether a label corresponds to Office or Residential
def OffRes(ans, lab):
    if lab[0] == "Office":
        if ans == 0:
            return 1
        else: 
            return 0
    else:
        if ans == 1:
            return 1
        else:
            return 0

# Determine the label for each day of the week
res = [0] * (len(labelsw))
for i in range(len(labelsw)):
    Off = 0
    Off += OffRes(labelsm[i], labm)
    Off += OffRes(labelstu[i], labtu)
    Off += OffRes(labelsw[i], labw)
    Off += OffRes(labelsth[i], labth)
    Off += OffRes(labelsf[i], labf)
    if Off > 2:
        res[i] = 'Office'
    else: 
        res[i] = 'Residential'

# Print the clustering results for each day
for i in range(len(labelsw)):
    print(str(i)+" - M: " + str(labelsm[i]) + ", Tu: " + str(labelstu[i]) + ", W: " + str(labelsw[i]) + ", Th: " + str(labelsth[i]) + ", F: " + str(labelsf[i]) + " - Result: " + str(res[i]))

# Create individual plots for each data point
for i in range(54):
    plt.figure(figsize=(26,14))
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.04, right=0.98, top=0.93, bottom=0.05)
    for k in range(int(20)):
        itr = 20 * i + k
        plt.subplot(4, 5, k+1)
        plt.plot(range(3), Xm[itr][:3], '.-', )
        plt.plot(range(3), Xtu[itr][:3], '.-', )
        plt.plot(range(3), Xw[itr][:3], '.-', )
        plt.plot(range(3), Xth[itr][:3], '.-', )
        plt.plot(range(3), Xf[itr][:3], '.-', )
        plt.ylim(0, 0.9)
        for ii in range(int(3)):
            plt.axvline(x=ii, color='black', linestyle=':')
        plt.grid('on', linestyle=':')
        plt.title(str(dfm.columns[itr])[:-14]+' - '+ str(res[itr]))
    plt.show()
