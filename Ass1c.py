import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
data = pd.read_excel (r'C:\Users\cleoo\Documents\Complex Network\manufacturing_emails_temporal_network.xlsx')
Gdata = data

#%% Create new data for G2 and G3
from random import shuffle

#Create G2
node1 = data['node1'].tolist(); node2 = data['node2'].tolist(); time = data['timestamp'].tolist()
shuffle(time) #shuffling lists is much faster than shuffling the dataframe
G2 = pd.DataFrame() #create dataframe for G2
G2['node1'] = node1
G2['node2'] = node2
G2['timestamp']= time
G2 = G2.sort_values(by='timestamp',ignore_index=True) #resort

#Creating G3
G = data.drop(['timestamp'],axis=1)
G = G.drop_duplicates() #G now contains all unique nodepairs
G = G.sort_values(by=['node1','node2'],ignore_index=True) #resort
node1 = G['node1'].tolist(); node2 = G['node2'].tolist(); time = data['timestamp'].tolist()
take_pair = np.random.randint(0,len(node1),len(time)) #create random array to select nodepair for each timestamp
new_node1 = np.zeros(len(time),dtype=int)
new_node2 = np.zeros(len(time),dtype=int)

for i in range(len(time)):
    new_node1[i] = node1[take_pair[i]]
    new_node2[i] = node2[take_pair[i]]
    
G3 = pd.DataFrame()
G3['node1'] = new_node1
G3['node2'] = new_node2
G3['timestamp']= time

#%% Find interarrival times
def arrival_times(df):
    df = df.sort_values(by=['node1','node2'],ignore_index=True) #resort
    node1 = df['node1'].tolist(); node2 = df['node2'].tolist(); time = df['timestamp'].tolist()
    time_diff = np.diff(time)
    t_arr = []
    for i in range(len(node1)-1):
        if ((node1[i+1]==node1[i]) and (node2[i+1]==node2[i])):
            t_arr.append(time_diff[i])
    return t_arr

t1 = arrival_times(Gdata)
t2 = arrival_times(G2)
t3 = arrival_times(G3)

#%% Plot the interarrival times in histograms
