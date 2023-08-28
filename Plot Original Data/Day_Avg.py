import pandas as pd
import matplotlib.pyplot as plt

# User choice for input settings
choice = 'n'
if choice == 'y':
    transfertype = str(input('Received (r) or Transferred (t): '))
    swc = int(input('Switch: '))
    time = int(input('Timestep: (Minutes) '))
    span = int(input('Span: (Hours) '))
    day = int(input('Day: (Monday - 0) ... (Sunday - y6) '))
    amount = int(input('Amount of weeks: '))
else:
    time = 20
    span = 24
    day = 0

# Map day index to day name
daylist = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# Map 'r' to 'Received' and 't' to 'Transferred'
transferdict = {'r': 'Received', 't': 'Transferred'}

# Read data and preprocess data
df = pd.read_feather("ifcounts_be25d1.feather")
df.set_index('index', inplace=True)
dropvalue = 7200 + (1440 * day)
df = df.drop(df.index[:dropvalue])
dfh = df.resample("" + str(time) + "Min").mean()  # Resample data at 'time' minute intervals
dfh.reset_index(inplace=True)  # Reset index

# Calculate the mean of a list of lists
def sum_list(avg):
    avglist = []
    for i in range(len(avg[0])):
        mean = sum(sublist[i] for sublist in avg) / len(avg)
        avglist.append(mean)

    return avglist

# Create a DataFrame with data
def listmaker(transfertype, amount):
    switchlist = pd.DataFrame()
    t = int(1440 / time)
    for switch in range(1, 7):
        for port in range(1, 25):
            if transfertype == 'r':
                column = dfh['be100kw' + str(switch) + '__eth' + str(port) + '__rxavgbitrate']
            elif transfertype == 't':
                column = dfh['be100kw' + str(switch) + '__eth' + str(port) + '__txavgbitrate']
            avg = []
            for i in range(amount):
                avg.append(list(column[t * (i * 7):t * ((i * 7) + 1)]))
            d = {'be100kw' + str(switch) + '__eth' + str(port) + '__avgbitrate__' + daylist[day]: sum_list(avg)}
            df = pd.DataFrame(data=d)

            switchlist = pd.concat([switchlist, df], axis=1)
    return switchlist

labelx = [19, 20, 21, 22, 23, 24]
labely = [1, 7, 13, 19]

# Create a list of lists containing overlapping data
def listmaker_overlap(transfertype, amount):
    t = int(1440 / time)
    overlap = []
    for switch in range(1, 7):
        for port in range(1, 25):
            if transfertype == 'r':
                column = dfh['be100kw' + str(switch) + '__eth' + str(port) + '__rxavgbitrate']
            elif transfertype == 't':
                column = dfh['be100kw' + str(switch) + '__eth' + str(port) + '__txavgbitrate']

            for i in range(amount):
                overlap.append(list(column[t * (i * 7):t * ((i * 7) + 1)]))
    return overlap

# Plot overlapping data
def plot_overlap(swc, eth, time, span, switchlist, amount):
    h = int(span * 60 / time)
    plt.subplot(4, 6, eth)
    
    for i in range(amount):
        print(swc, eth, i)
        plt.plot(range(1, h + 1), switchlist[(swc - 1) * 24 * amount + (eth - 1) * amount + i], '-', marker='.', markevery=int(60 / time))
    
    plt.title('Port: ' + str(eth))
    if eth in labelx:
        plt.xlabel(str(time) + ' Minute Interval')
    if eth in labely:
        plt.ylabel('Average bitrate, Day: ' + daylist[day])
    
    for i in range(1, int(span / 8)):
        plt.axvline(x=8 * i * (60 / time), color='black', linestyle=':')
    for i in range(1, int(span / 24)):
        plt.axvline(x=24 * i * (60 / time), color='red', linestyle=':')

    plt.xlim([1, h])
    plt.grid('on', linestyle=':')

# Plot the non-overlapping data
def plot(swc, eth, time, span, switchlist):
    h = int(span * 60 / time)
    plt.subplot(4, 6, eth)
    plt.plot(range(1, h + 1), list(switchlist['be100kw' + str(swc) + '__eth' + str(eth) + '__avgbitrate__' + daylist[day]].head(h)), '-', color='tab:blue', marker='.', markevery=int(60 / time))
    plt.title('Port: ' + str(eth))
    if eth in labelx:
        plt.xlabel(str(time) + ' Minute Interval')
    if eth in labely:
        plt.ylabel('Average bitrate, Day: ' + daylist[day])
    
    for i in range(1, int(span / 8)):
        plt.axvline(x=8 * i * (60 / time), color='black', linestyle=':')
    for i in range(1, int(span / 24)):
        plt.axvline(x=24 * i * (60 / time), color='red', linestyle=':')

    plt.xlim([1, h])
    plt.grid('on', linestyle=':')

# Run the plotting process
def run(swc, time, span, transfertype, day, amount, overlap):
    plt.figure(figsize=(26, 14))
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.04, right=0.98, top=0.93, bottom=0.05)
    plt.suptitle(transferdict[transfertype] + ' average bitrate')
    
    if overlap:
        switchlist = listmaker_overlap(transfertype, amount)
        for i in range(1, 25):
            plot_overlap(swc, i, time, span, switchlist, amount)
        plt.savefig('Switch' + str(swc) + '_' + transfertype + 'bavg_' + str(time) + '_allports_' + daylist[day] + '__' + str(amount) + '_overlap.png')
    else:
        switchlist = listmaker(transfertype, amount)
        for i in range(1, 25):
            plot(swc, i, time, span, switchlist)
        plt.savefig('Switch' + str(swc) + '_' + transfertype + 'bavg_' + str(time) + '_allports_' + daylist[day] + '__' + str(amount) + '.png')

# Run code with specifications
if choice == 'y':
    run(swc, time, span, transfertype, day, amount, True)
else:
    for i in range(7):
        run(i, 20, 24, 'r', 0, 7, True)
        run(i, 20, 24, 't', 0, 7, True)