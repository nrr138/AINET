import pandas as pd

# Load Office data
df1 = pd.read_feather("Office_1.feather")
df2 = pd.read_feather("Office_2.feather")
df1.set_index('index', inplace=True)
df2.set_index('index', inplace=True)

# Concatenate Office dataframes horizontally
df = pd.concat([df1, df2], axis=1)
df.reset_index(inplace=True)

# Save concatenated Office dataframe to a new feather file
df.to_feather('OfficeAll.feather')

# Load Residential data
df1 = pd.read_feather("Residential_1.feather")
df2 = pd.read_feather("Residential_2.feather")
df1.set_index('index', inplace=True)
df2.set_index('index', inplace=True)

# Concatenate Residential dataframes horizontally
df = pd.concat([df1, df2], axis=1)
df.reset_index(inplace=True)


# Save concatenated Residential dataframe to a new feather file
df.to_feather('ResidentialAll.feather')
