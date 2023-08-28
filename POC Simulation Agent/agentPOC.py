import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator 

# Clean the DataFrame
def clean_feather(df):
    df.set_index('index', inplace=True)
    columns = [col for col in df.columns if col.startswith('be100kw1__eth10')]
    df = df[columns]
    df = df.resample(""+str(time)+"Min").mean() # Take mean of "time" minute intervals
    df = df.reindex(sorted(df.columns), axis=1)
    return df

# Load the original data into a pandas dataframe
time = 20
data = pd.read_feather('Data/ResidentialAll.feather')
df = clean_feather(data)

# Calculate the total network usage from the original data
columns = [col for col in df.columns if col.endswith('txavgbitrate') or col.endswith('rxavgbitrate')]
dfh = df[columns]
total_usage = dfh.sum().sum()

# Define the parameters for the agents
num_agents_per_port = 4
initial_usage_ratio = 1
max_usage_change = 0.9

# Create a list of agents
agents = []
for i in range(int(len(df.columns)/8)):
    port_data = df.iloc[:, i*8:(i+1)*8]
    port_usage = port_data.iloc[:,6:8].sum().sum()
    for _ in range(num_agents_per_port):
        initial_usage = initial_usage_ratio * (port_usage.sum() / num_agents_per_port)
        agents.append({'port': i, 'usage': [initial_usage] * len(port_data.index), 'minute': port_data.index})

# Run the simulation loop for each minute and update the usage of each agent
for m in range(len(df.index)):
    for agent in agents:
        usage_change = np.random.uniform(-max_usage_change, max_usage_change) * agent['usage'][m]
        agent['usage'][m] += usage_change
        if agent['usage'][m] < 0:
            agent['usage'][m] = 0

# Create a new dataframe with the updated network usage of each agent for each minute
simulated_data = pd.DataFrame(columns=['Port', 'Agent', 'Minute', 'Received Bitrate', 'Transferred Bitrate'])
for i in range(int(len(df.columns)/8)):
    port_data = df.iloc[:, i*8:(i+1)*8]
    agent_num = 0
    for agent in agents:
        if agent['port'] == i:
            new_rows = {
                'Port': i,
                'Agent': agent_num,
                'Minute': agent['minute'],
                'Received Bitrate': port_data.iloc[:,7] * (agent['usage'] / port_data.iloc[:,7].sum(axis=0)),
                'Transferred Bitrate': port_data.iloc[:,6] * (agent['usage'] / port_data.iloc[:,6].sum(axis=0)),
                }
            new_df = pd.DataFrame(new_rows)
            simulated_data = pd.concat([simulated_data, new_df])
            agent_num +=1 

# Plot original and simulated data
def plot_compare(star, fin, T):
    port = simulated_data.loc[simulated_data['Port'] == 0]
    plot_sum = [0]*T
    for i in range(num_agents_per_port):
        age = port.loc[port['Agent'] == i]['Received Bitrate']
        plot = [0]*T
        for i in range(star,fin):
            plot = list(map(operator.add, list(age[i*T:(i+1)*T]), plot))
        plot[:] = [x /(fin-star) for x in plot]
        plt.plot(range(len(plot)), plot, '-')
        plot_sum = list(map(operator.add,plot,plot_sum))
    ploto = [0]*T
    o = df.iloc[:, 7]
    for i in range(star,fin):
        ploto = list(map(operator.add, list(o[i*T:(i+1)*T]), ploto))
    ploto[:] = [x /(fin-star) for x in ploto]

    plt.plot(range(len(plot_sum)), plot_sum, '-', color= 'tab:purple')
    plt.plot(range(len(ploto)), ploto, '-', color= 'black', marker='.')
    for i in range(1, 3):
        plt.axvline(x=i*(len(ploto)/3), color='black', linestyle=':')
        
    plt.legend(['Agent 1', 'Agent 2', 'Agent 3','Agent 4', 'Simulated', 'Original'])
    plt.xlabel('Time')
    plt.ylabel('Received bitrate')
    plt.grid('on', linestyle=':')
    plt.show()

# Call plot function
plot_compare(0,100, int(24*60/time))
for i in range(100):
    plot_compare(i,i+1,int(24*60/time))
