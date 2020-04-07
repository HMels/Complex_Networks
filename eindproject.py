# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import igraph as igraph
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import random

import numpy.random as rnd
import timeit

simulation = 'Haggle' 
#simulation = 'MIT'

if simulation == 'Haggle':
    B = pd.read_excel (r'MIT_data_sorted.xlsx')
if simulation == 'MIT':
    B = pd.read_excel (r'data_Haggle_sorted.xlsx')

#data = data.drop_duplicates()

data = B
Nnodes = np.max([data['node1'].max(), data['node2'].max()])
Nlinks = len(data)
Tmin = np.max([data['timestamp'].min()])
if simulation == 'Haggle':
    data['timestamp'] = ( data['timestamp'] - Tmin)  
if simulation == 'MIT':
    data['timestamp'] = ( data['timestamp'] - Tmin)/600         #first timestamp is 0, every timestep is 10 minutes

#%% Make iGraph
g = igraph.Graph()
g.add_vertices(Nnodes)
col1 = data['node1']; col2 = data['node2']; col3 = data['timestamp']
col1 = col1.tolist(); col2 = col2.tolist(); col3 = col3.tolist()
col1 = col1[:int(Nlinks/3)]; col2 = col2[:int(Nlinks/3)]; col3 = col3[:int(Nlinks/3)]
#%%
Nlinks = len(col1)
iets=0.001

for i in range(Nlinks):
    
    g.add_edges([(col1[i]-1,col2[i]-1)])
    p = rnd.rand()
    if p<iets: 
        a = [ str(np.round(i/Nlinks*100,2)) + ' % ']
        clear_output(wait=True)
        display(a)
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
#%%
Adj = g.get_adjacency()
Adj = np.array([Adj.data])
Eig = np.linalg.eig(Adj)
MaxEig = np.max(Eig[0])

Lap = g.laplacian()
Lap = np.array([Lap])
Eiglap = np.linalg.eig(Lap) #stores eigenvalues and eigenvectors
Alg_con = np.sort(Eiglap[0])
Alg_con = Alg_con[:,1] #take second smallest eigenvalue

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
"""Evaluation starts here"""
start = timeit.default_timer()
tmax = int(data.timestamp.max())
beta = 0.2
if simulation == 'Haggle':
    gamma = 0.0002/600
if simulation == 'MIT':
    gamma = 0.0002
Infections = np.zeros([tmax,Nnodes])
Removed = np.zeros([Nnodes, Nnodes])
Removed_total = np.zeros([tmax,Nnodes])
Susceptible = np.zeros([tmax,Nnodes])
data_dropped = data

#%% Totaal random
n_removed = round(len(data)*0.995)
delete_row = random.sample(range(len(data)),n_removed)
data_dropped = data_dropped.drop(delete_row)
##delete_row.index(1048575)
#%% Gebaseerd op minst gebruikte links
duplicates = data.pivot_table(index=['node1','node2'], aggfunc='size')
duplicates = pd.Series.sort_values(duplicates,ascending=False)

n_deleted_links = 50000

som = 0
for i in range(len(duplicates)):
    som = duplicates.values[-i] + som
    if som > n_deleted_links:
        row_stop = i
        print('Number of rows to delete:', i)
        break

for i in range(row_stop):
    drop_indices = data[(data[['node1','node2']] == duplicates.index[-i]).all(1)].index.tolist()
    data_dropped = data_dropped.drop(drop_indices)
    if i  % 100 == 0:
        print('We are at:', round(i/row_stop*100), '%. Elapsed Time', round(timeit.default_timer()-start))
print(len(data)-len(data_dropped), 'links are deleted')
#%% Gebaseerd op x aantal keer dat een link mag voorkomen

Lmax = 2

for i in range(len(duplicates)):
    drop_indices = data[(data[['node1','node2']] == duplicates.index[-i]).all(1)].index#.tolist()
    if len(drop_indices) > Lmax:
        delete_row = random.sample(range(len(drop_indices)),len(drop_indices)-Lmax)
        delete_row = np.array(delete_row)
        data_dropped = data_dropped.drop(drop_indices[delete_row])
    if i  % 100 == 0:
        print('We are at:', round(i/len(duplicates)*100), '%')

print(len(data)-len(data_dropped), 'links are deleted')


#%%

Aoud = np.eye(Nnodes)
unit = np.eye(Nnodes)


stop = timeit.default_timer()
print('Starting evaluation,elapsed time', round(stop-start))

for i in range(0,tmax):
    #data_temp = data[data.timestamp==i].values
    data_temp = data_dropped[data_dropped.timestamp==i].values
    A = np.zeros([Nnodes,Nnodes])
    w = int(len(data_temp))
    
    if data_temp.size:
        for j in range(w):
            p = rnd.rand()
            if p<beta:    # When p is smaller than beta, (0.11<0.2) then the contact will be counted as an infection, if not, no infection so no changes in infection matrix
                A[int(data_temp[j,0]-1),int(data_temp[j,1]-1)] = 1
                A[int(data_temp[j,1]-1),int(data_temp[j,0]-1)] = 1
      
        Inf = np.dot(A+unit,Aoud) #infectable content
        Inf[Inf>0]=1
        Aoud = Inf - Removed #current infected nodes
        Aoud[Aoud<0]=0
    if True:
        locate = np.argwhere(Aoud>0) #search for infected nodes
    
        for l in locate: #removed nodes, probability to recover is gamma
            q = rnd.rand()
            if q<gamma:
                Aoud[l[0],l[1]] = 0
                Removed[l[0],l[1]] = 1 #Stores which nodes are immune
            
#            for l in range(len(Aoud[0,:])):
#        for k in range(len(Aoud[:,0])):
#            q = rnd.rand()
#            if q<gamma:
#                Removed[l,k] = Aoud[l,k]
#                Aoud[l,k] = 0          
    Removed_total[i,:] = np.sum(Removed, axis=0)      
    Infections[i,:] = np.sum(Aoud, axis=0)
    Susceptible[i,:] = Nnodes - Removed_total[i,:] - Infections[i,:]
    if i  % 1000 == 0:
        print('We are at:', round(i/tmax*100), '%')
    
stop = timeit.default_timer()
print('Elapsed Time:',stop-start)


#%%

plt.close("all")

ExpVal = np.sum(Infections, axis = 1)/Nnodes
StandardDev = np.std(Infections, axis = 1)

t=np.linspace(1,tmax,len(ExpVal))
ExpVal_rem = np.sum(Removed_total, axis = 1)/Nnodes
StandardDev_rem = np.std(Removed_total, axis = 1)

y1 = ExpVal
y2 = Nnodes - ExpVal_rem
if simulation == 'Haggle':
    x=t/60/60/24  
if simulation == 'MIT':
    x=t/6/24 

#plt.title('Average Infected Nodes Versus Time With Corresponding Standard Deviation')
plt.figure()
plt.xlabel('Time(days)')
plt.ylabel('Average Infected Nodes')
plt.axes(ylim=(0,Nnodes),xlim=(0,np.max(x)))

plt.errorbar(x,ExpVal,yerr = StandardDev, errorevery = 400, ecolor = 'r', color = 'k')
plt.errorbar(x,Nnodes - ExpVal_rem,yerr = StandardDev_rem, errorevery = 452, ecolor = 'y', color = 'b')

plt.figure()
plt.plot(x,y1,'k',x,y2,'k')
plt.axes(ylim=(0,Nnodes),xlim=(0,np.max(x)))
plt.fill_between(x,y1,y2,where=y2>=y1,facecolor = 'teal')
plt.fill_between(x,0,y1,facecolor = 'r')
plt.fill_between(x,y2,Nnodes,facecolor = 'silver')


ExpVal_inf = np.sum(Infections, axis = 1)/Nnodes
ExpVal_sus = np.sum(Susceptible, axis = 1)/Nnodes
ExpVal_rem2 = np.sum(Removed_total, axis = 1)/Nnodes

t=np.linspace(1,tmax,len(ExpVal_inf))
plt.figure()
plt.plot(x,ExpVal_inf, 'k')
plt.plot(x,ExpVal_sus, 'b')
plt.plot(x,ExpVal_rem2, 'r')
plt.legend(('Infected' ,'Susceptible', 'Removed'))








