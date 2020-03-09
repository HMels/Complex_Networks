#Ass1B
import numpy as np
import pandas as pd
import igraph
import timeit
import matplotlib.pyplot as plt
from random import shuffle

data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
#data = pd.read_excel (r'C:\Users\cleoo\Documents\Complex Network\manufacturing_emails_temporal_network.xlsx')

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
G3 = G3.drop_duplicates()

#%% Initialize matrices
Nd = len(G3)
Nt = 57791
Nnodes = 167
#inf_node = 13; #Again: node 1 is index 0!! 
num_inf = np.zeros([Nt,Nnodes]) #number of infected nodes
data = G3

#%%
# start = timeit.default_timer()
# for k in range(Nnodes):
#     inf_node = k+1
#     infection = [False]*167 
#     infection[inf_node-1] = True #true if a node is infected
#     for i in range(Nt):
#         TSdata = data.iloc[data.index[data['timestamp'] == (i+1)]]
#         tot_inf = np.where(infection)[0]+1
#         for j in range(len(TSdata.index)):
#             if TSdata.iloc[j,0] in tot_inf:
#                 infection[TSdata.iloc[j,1]-1] = True
#             if TSdata.iloc[j,1] in tot_inf:
#                 infection[TSdata.iloc[j,0]-1] = True
#         num_inf[i,k] = sum(infection)
#     print(k)
# stop = timeit.default_timer()
# print('Time: ', stop - start)

#%%
start = timeit.default_timer()
for k in range(Nnodes):
    inf_node = k+1
    infection = [False]*167 
    infection[inf_node-1] = True #true if a node is infected
    tot_inf = np.where(infection)[0]+1
    index = 1
    for i in range(Nd):
        if index == data.iloc[i,2]:
            if data.iloc[i,0] in tot_inf:
                infection[data.iloc[i,1]-1] = True
            if data.iloc[i,1] in tot_inf:
                infection[data.iloc[i,0]-1] = True
        else:
            tot_inf = np.where(infection)[0]+1
            num_inf[index,k] = sum(infection)
            index = index + 1
            if data.iloc[i,0] in tot_inf:
                infection[data.iloc[i,1]-1] = True
            if data.iloc[i,1] in tot_inf:
                infection[data.iloc[i,0]-1] = True       
    print(k)
    stop = timeit.default_timer()
    print('Time: ', stop - start)