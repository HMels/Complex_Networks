# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:53:49 2020

@author: stijn
"""

import numpy as np
import pandas as pd
import igraph as igraph
import timeit
import matplotlib.pyplot as plt

#data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
data = pd.read_excel (r'..\manufacturing_emails_temporal_network.xlsx')
#data = pd.read_excel (r'C:\Users\rixtb\Documents\Master\Data analysis\Datasets\oefenset.xlsx')


#%% A
Nnodes = np.max([data['node1'].max(), data['node2'].max()])
G = data.drop(['timestamp'],axis=1)
B = G.drop_duplicates()
g = igraph.Graph()
g.add_vertices(Nnodes)

col1 = B['node1']
col2 = B['node2']

col1 = col1.tolist()
col2 = col2.tolist()

Nlinks = len(B)
for i in range(Nlinks):
    g.add_edges([(col1[i]-1,col2[i]-1)])
    

#%% B
tmax = data.timestamp.max()
Infections = np.zeros([tmax,Nnodes])

start = timeit.default_timer()

Aoud = np.eye(Nnodes)
unit = np.eye(Nnodes)

for i in range(1,tmax+1):
    data_temp = data[data.timestamp==i].values
    A = np.zeros([Nnodes,Nnodes])
    
    for j in range(len(data_temp)):
        A[data_temp[j,0]-1,data_temp[j,1]-1] = 1
        A[data_temp[j,1]-1,data_temp[j,0]-1] = 1
    
    Inf = np.dot(A+unit,Aoud)
    Inf[Inf>0]=1
    Aoud = Inf
    Infections[i-1,:] = np.sum(Inf, axis=0)
    
stop = timeit.default_timer()
print('Time:',stop-start)
#%% 9
plt.close("all")

ExpVal = np.sum(Infections, axis = 1)/Nnodes
StandardDev = np.std(Infections, axis = 1)

t=np.linspace(1,tmax,len(ExpVal))
plt.axes(xlim=(1,tmax))
plt.xlabel('Time(s)')
plt.ylabel('Average Infected Nodes')
plt.title('Average Infected Nodes Versus Time With Corresponding Standard Deviation')
plt.errorbar(t,ExpVal,yerr = StandardDev, errorevery = 100, ecolor = 'r', color = 'k')
#%% 10
R = np.ones(Nnodes)*float('nan')

for k in range(Nnodes):
    for i in range(tmax):
       if Infections[i,k] > 0.8*Nnodes:
           R[k] = i+1 
           break
#%%11
           
C2 = np.zeros(Nnodes)
D2 = np.zeros(Nnodes)

C = g.transitivity_local_undirected(vertices = None, mode = "zero")
C_ind = np.argsort(C)[::-1]+1
C.sort(reverse=True)
D = g.outdegree()
D_ind = np.argsort(D)[::-1]+1
D.sort(reverse=True)
    

f = np.linspace(0.05,0.5,10)
for i in f:
    print(i)
    R_f =round(i*Nnodes)
    
    rd_f = 
