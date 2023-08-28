import pandas as pd
import matplotlib.pyplot as plt

# User choice for input settings
choice = 'n' 
if choice == 'y':
    transfertype = str(input('Received (r) or Transferred (t): '))
    swc = int(input('Switch: '))
    time = int(input('Timestep: (Minutes) '))
    span = int(input('Span: (Hours) '))
else:
    time = 20
    span = 24

# Map 'r' to 'Received' and 't' to 'Transferred'
transferdict = {'r': 'Received', 't': 'Transferred'}

# Read and preprocess data
df = pd.read_feather("ifcounts_be25d1.feather")
df.set_index('index', inplace=True)
df = df.drop(df.index[:7200])
dfh = df.resample("" + str(time) + "Min").mean()  # Resample data at 'time' minute intervals
dfh.reset_index(inplace=True)  # Reset index

# Calculate the mean of a list of lists
def sum_list(avg):
    avglist = []

    for i in range(len(avg[0])):
        mean = sum(sublist[i] for sublist in avg) / len(avg)
        avglist.append(mean)

    return avglist

# Create DataFrame 
def listmaker(transfertype):
    switchlist = pd.DataFrame()
    t = int(1440 / time)
    for switch in range(1, 7):
        for port in range(1, 25):
            if transfertype == 'r':
                column = dfh['be100kw' + str(switch) + '__eth' + str(port) + '__rxavgbitrate']
            elif transfertype == 't':
                column = dfh['be100kw' + str(switch) + '__eth' + str(port) + '__txavgbitrate']
            avg_v = []
            avg_h = []
            for i in range(5):
                avg_v.append(list(column[t * i:t * (i + 1)]))
            for i in range(2):
                avg_h.append(list(column[t * (i + 5):t * (i + 6)]))
            d = {'be100kw' + str(switch) + '__eth' + str(port) + '__avgbitrate__weekdays': sum_list(avg_v),
                 'be100kw' + str(switch) + '__eth' + str(port) + '__avgbitrate__weekend': sum_list(avg_h)}
            df = pd.DataFrame(data=d)

            switchlist = pd.concat([switchlist, df], axis=1)
    return switchlist

labelx = [19, 20, 21, 22, 23, 24]
labely = [1, 7, 13, 19]

# Plot the data
def plot(swc, eth, time, span, switchlist):
    h = int(span * 60 / time)
    plt.subplot(4, 6, eth)
    plt.plot(range(1, h + 1),
             list(switchlist['be100kw' + str(swc) + '__eth' + str(eth) + '__avgbitrate__weekdays'].head(h)),
             '-', color='tab:blue', marker='.', markevery=int(60 / time))
    plt.plot(range(1, h + 1),
             list(switchlist['be100kw' + str(swc) + '__eth' + str(eth) + '__avgbitrate__weekend'].head(h)),
             '-', color='tab:orange', marker='.', markevery=int(60 / time))
    plt.title('Port: ' + str(eth))
    if eth in labelx:
        plt.xlabel(str(time) + ' Minute Interval')
    if eth in labely:
        plt.ylabel('Average bitrate')

    for i in range(1, int(span / 8)):
        plt.axvline(x=8 * i * (60 / time), color='black', linestyle=':')
    for i in range(1, int(span / 24)):
        plt.axvline(x=24 * i * (60 / time), color='red', linestyle=':')

    plt.xlim([1, h])
    plt.grid('on', linestyle=':')

# Run the plotting process
def run(swc, time, span, transfertype):
    plt.figure(figsize=(26, 14))
    plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.04, right=0.98, top=0.93, bottom=0.05)
    plt.suptitle('Average bitrate (blue=weekdays, orange=weekends) ' + transferdict[transfertype])
    switchlist = listmaker(transfertype)
    for i in range(1, 25):
        plot(swc, i, time, span, switchlist)
    plt.savefig('Switch' + str(swc) + '_' + transfertype + 'bavg_' + str(time) + '_allports_weekend-days.png')

# Run code
if choice == 'y':
    run(swc, time, span, transfertype)
else:
    for i in range(1, 7):
        run(i, 20, 24, 'r')
        run(i, 20, 24, 't')
