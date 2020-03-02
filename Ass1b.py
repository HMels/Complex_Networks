#Ass1B
import numpy as np
import pandas as pd
import igraph
import timeit
import matplotlib.pyplot as plt

data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
#data = pd.read_excel (r'C:\Users\cleoo\Documents\Complex Network\manufacturing_emails_temporal_network.xlsx')

#%% Initialize matrices
Nt = 57791
Nnodes = 2
#inf_node = 13; #Again: node 1 is index 0!! 
num_inf = np.zeros([Nt,Nnodes]) #number of infected nodes

#%%
start = timeit.default_timer()
for k in range(Nnodes):
    inf_node = k+1
    infection = [False]*167 
    infection[inf_node-1] = True #true if a node is infected
    for i in range(Nt):
        TSdata = data.iloc[data.index[data['timestamp'] == (i+1)]]
        tot_inf = np.where(infection)[0]+1
        for j in range(len(TSdata.index)):
            if TSdata.iloc[j,0] in tot_inf:
                infection[TSdata.iloc[j,1]-1] = True
            if TSdata.iloc[j,1] in tot_inf:
                infection[TSdata.iloc[j,0]-1] = True
        num_inf[i,k] = sum(infection)
    print(k)
stop = timeit.default_timer()
print('Time: ', stop - start)