# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import igraph as igraph
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


#data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
B = pd.read_excel (r'C:\Users\rixtb\Documents\Master\Data analysis\Datasets\MIT_data_sorted.xlsx')
#data = pd.read_excel (r'C:\Users\rixtb\Documents\Master\Data analysis\Datasets\oefenset.xlsx')
#data = data.drop_duplicates()

#%% 
data = B
Nnodes = np.max([data['node1'].max(), data['node2'].max()])
Nlinks = len(data)
Tmin = np.max([data['timestamp'].min()])
data['timestamp'] = ( data['timestamp'] - Tmin) /600         #first timestamp is 0, every timestep is 10 minutes
g = igraph.Graph()
g.add_vertices(Nnodes)
col1 = data['node1']; col2 = data['node2']; col3 = data['timestamp']
col1 = col1.tolist(); col2 = col2.tolist(); col3 = col3.tolist()
col1 = col1[:int(Nlinks/3)]; col2 = col2[:int(Nlinks/3)]; col3 =col3[:int(Nlinks/3)]
#%%
Nlinks = len(col1)

for i in range(Nlinks):
    
    g.add_edges([(col1[i]-1,col2[i]-1)])
#    a = [ str(np.round(i/Nlinks*100,2)) + ' % ']
#    clear_output(wait=True)
#    display(a)
#%% Properties temporal network
ED = g.degree()
p = g.density(loops=False) #ratio between the actual links and the possible edges
average_degree = 2* Nlinks/Nnodes #see slides lecture 1
var = np.var(ED)
plt.hist(ED,bins=Nnodes)
plt.plot()
pd = g.assortativity_degree()
clust_coef = g.transitivity_undirected()
average_path_length=g.average_path_length()
hopcount_max = np.max(g.shortest_paths())

#%% Calculate temporal network
data_ag = B
data_ag = data_ag.drop(['timestamp'],axis=1)
data_ag = data_ag.drop_duplicates()
Nnodes_ag = np.max([data_ag['node1'].max(), data_ag['node2'].max()])
Nlinks_ag = len(data_ag)
#Tmin2 = np.max([data2['timestamp'].min()])
#data2['timestamp'] = data2['timestamp'] - Tmin2             #first timestamp is 0
g_ag = igraph.Graph()
g_ag.add_vertices(Nnodes_ag)
col1_ag = data_ag['node1'].tolist(); col2_ag = data_ag['node2'].tolist();# col32 = data2['timestamp']

#%%
Nlinks_ag = len(col1_ag)
for i in range(Nlinks_ag):
    
    g_ag.add_edges([(col1_ag[i]-1,col2_ag[i]-1)])
#    a = [ str(np.round(i/Nlinks2*100,2)) + ' % ']
#    clear_output(wait=True)
#    display(a)

#%% Properties of temporal network
# These properties can be compared with http://konect.uni-koblenz.de/networks/mit , should be the same
ED_ag = g_ag.degree()
p_ag = g_ag.density(loops=False) #ratio between the actual links and the possible edges
average_degree_ag = 2* Nlinks_ag/Nnodes_ag #see slides lecture 1
var_ag = np.var(ED_ag)
plt.hist(ED_ag,bins=Nnodes_ag)
plt.plot()
pd_ag = g_ag.assortativity_degree()
clust_coef_ag = g_ag.transitivity_undirected()
average_path_length_ag=g_ag.average_path_length()
hopcount_max_ag = np.max(g_ag.shortest_paths())

#%%
Adj_ag = g_ag.get_adjacency()
Adj_ag = np.array([Adj_ag.data])
Eig_ag = np.linalg.eig(Adj_ag)
MaxEig_ag = np.max(Eig_ag[0])

Lap_ag = g_ag.laplacian()
Lap_ag = np.array([Lap_ag])
Eiglap_ag = np.linalg.eig(Lap_ag) #stores eigenvalues and eigenvectors
Alg_con_ag = np.sort(Eiglap_ag[0])
Alg_con_ag = Alg_con_ag[:,1] #take second smallest eigenvalue

#%% Infection of network using adjancency matrix

import numpy.random as rnd
import timeit
#%%
tmax = int(data.timestamp.max())
beta = 0.2
gamma = 
Infections = np.zeros([tmax,Nnodes])

start = timeit.default_timer()

Aoud = np.eye(Nnodes)
unit = np.eye(Nnodes)

for i in range(0,tmax):
    data_temp = data[data.timestamp==i].values
    A = np.zeros([Nnodes,Nnodes])
    w = int(len(data_temp))
    
    
    for j in range(w):
        p = rnd.rand()
        if p<beta:    # When p is smaller than beta, (0.11<0.2) then the contact will be counted as an infection, if not, no infection so no changes in infection matrix
            A[int(data_temp[j,0]-1),int(data_temp[j,1]-1)] = 1
            A[int(data_temp[j,1]-1),int(data_temp[j,0]-1)] = 1
    
    Inf = np.dot(A+unit,Aoud)
    Inf[Inf>0]=1
    Aoud = Inf
    Infections[i-1,:] = np.sum(Inf, axis=0)
        
stop = timeit.default_timer()
print('Time:',stop-start)

#%%

plt.close("all")

ExpVal = np.sum(Infections, axis = 1)/Nnodes
StandardDev = np.std(Infections, axis = 1)

t=np.linspace(1,tmax,len(ExpVal))
plt.axes(xlim=(1,tmax))
plt.xlabel('Time(s)')
plt.ylabel('Average Infected Nodes')
#plt.title('Average Infected Nodes Versus Time With Corresponding Standard Deviation')
plt.errorbar(t,ExpVal,yerr = StandardDev, errorevery = 100, ecolor = 'r', color = 'k')














