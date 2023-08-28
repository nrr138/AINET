import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
# Clean and preprocess data
def clean_feather(df):
    time = 480 
    df.set_index('index', inplace=True)
    df = df.drop(df.index[:7200])
    columns = [col for col in df.columns if col.endswith('txavgbitrate')]
    dfh = df[columns]
    dfh = dfh.resample(""+str(time)+"Min").mean() # Take mean of "time" minute intervals
    column_sums = dfh.sum(axis=0)
    for i in [col for col in column_sums.index if column_sums[col] < 100000]:
        dfh = dfh[dfh.columns.drop(i)]
    dfh = dfh.fillna(0)
    return dfh

# Load and clean data
df1 = clean_feather(pd.read_feather("ifcounts_be25d1.feather"))
df2 = clean_feather(pd.read_feather("ifcounts_sh1d1.feather"))
df3 = clean_feather(pd.read_feather("ifcounts_ca3d1.feather"))
df4 = clean_feather(pd.read_feather("ifcounts_po9d1.feather"))
df5 = clean_feather(pd.read_feather("ifcounts_bs3d1.feather"))
df6 = clean_feather(pd.read_feather("ifcounts_br6d1.feather"))
df7 = clean_feather(pd.read_feather("ifcounts_br1d1.feather"))

# Concatenate cleaned dataframes
df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1)
df.reset_index(inplace=True) # Reset index 
df = df.drop('index', axis=1)
db = df

# Accumulate data for each day of the week
df_t = pd.DataFrame()
for i in df:
    save_days_t = [0]*21
    for ind in range(len(df[i])):
        save_days_t[ind%21] = save_days_t[ind%21]+ df[i][ind]

    df_t[i] = save_days_t

# Split the data for each day of the week
df_mon = pd.DataFrame()
df_tue = pd.DataFrame()
df_wed = pd.DataFrame()
df_thu = pd.DataFrame()
df_fri = pd.DataFrame()

for i in df_t:
    df_mon[i] = [df_t[i][j] for j in range(3)]
    df_tue[i] = [df_t[i][j+3] for j in range(3)]
    df_wed[i] = [df_t[i][j+6] for j in range(3)]
    df_thu[i] = [df_t[i][j+9] for j in range(3)]
    df_fri[i] = [df_t[i][j+12] for j in range(3)]

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

# Fit the model to the data
kmeansm.fit(Xm)
kmeanstu.fit(Xtu)
kmeansw.fit(Xw)
kmeansth.fit(Xth)
kmeansf.fit(Xf)

# Predict the cluster labels for each data point
labelsm = kmeansw.fit_predict(Xm)
labelstu = kmeansw.fit_predict(Xtu)
labelsw = kmeansw.fit_predict(Xw)
labelsth = kmeansw.fit_predict(Xth)
labelsf = kmeansw.fit_predict(Xf)

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
res = [0]*(len(labelsw))
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

# Function to prepare data for saving
def prep(df):
    df.set_index('index', inplace=True)
    dfh = df.filter(regex='(rx|tx)avgbitrate|pkts(64|65_127|128_255|256_511|512_1023|1024_max)')
    return dfh

# Prepare and save Office and Residential data
d1 = prep(pd.read_feather("ifcounts_be25d1.feather"))
d2 = prep(pd.read_feather("ifcounts_sh1d1.feather"))
d3 = prep(pd.read_feather("ifcounts_ca3d1.feather"))
#d4 = prep(pd.read_feather("ifcounts_po9d1.feather"))
#d5 = prep(pd.read_feather("ifcounts_bs3d1.feather"))
#d6 = prep(pd.read_feather("ifcounts_br6d1.feather"))
d7 = prep(pd.read_feather("ifcounts_br1d1.feather"))

dk = pd.concat([d1, d2, d3, d7], axis=1)
#dk = pd.concat([d4, d5, d6], axis=1)

Olst = []
Rlst = []
od = []
rd = []

for i in range(len(res)):
    if res[i] == 'Office':
        Olst.append(dfm.columns[i][:-12])
    elif res[i] == 'Residential':
        Rlst.append(dfm.columns[i][:-12])

# Select Office and Residential columns based on classification
do = dk[dk.columns[dk.columns.str.startswith(tuple(Olst))]]
dr = dk[dk.columns[dk.columns.str.startswith(tuple(Rlst))]]

# Reset index and save data to feather files
dr.reset_index(inplace=True)
do.reset_index(inplace=True)
do.to_feather('Office_2.feather')
dr.to_feather('Residential_2.feather')

