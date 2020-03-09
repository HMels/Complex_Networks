# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:31:15 2020

@author: stijn
"""

import numpy as np
import pandas as pd
import igraph as igraph
import timeit
import matplotlib.pyplot as plt

data = pd.read_excel (r'..\manufacturing_emails_temporal_network.xlsx')
data = data.drop_duplicates()
"""Create G2 by randomly shuffling the rows of the data set and resetting the indices"""
data['timestamp'] = data['timestamp'].sample(frac=1,random_state = 10).reset_index(drop=True)
data = data.sort_values(by='timestamp', axis =0 )
data = data.reset_index(drop=True)

#data = pd.read_excel (r'C:\Users\rixtb\Documents\Master\Data analysis\Datasets\oefenset.xlsx')

#%% A
start = timeit.default_timer()

Nnodes = np.max([data['node1'].max(), data['node2'].max()])
G = data.drop(['timestamp'],axis=1)
B = G.drop_duplicates()
g = igraph.Graph()
g.add_vertices(Nnodes)

col1 = B['node1']
col2 = B['node2']
#col3 = data['timestamp']

col1 = col1.tolist()
col2 = col2.tolist()
#col3 = col3.tolist()

Nlinks = len(B)
for i in range(Nlinks):
    g.add_edges([(col1[i]-1,col2[i]-1)])
    
stop = timeit.default_timer()
print('Time:',stop-start)

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

#%% creating the inter-arrival time histogram

BB=np.round(np.linspace(0,len(B)-1,len(B)),0)
B1 = B.assign(i=BB)

GG = pd.merge(data, B1, on = ['node1','node2'], how='left')        
#GG2 = np.zeros([len(GG['node1']),4])
GG = np.array([GG['node1'].tolist(),GG['node2'].tolist(),GG['timestamp'].tolist(),GG['i'].tolist()]).T

dT = []
for i in range(len(B)):
    delta_t = np.array([GG[GG[:,3]==i][:,2]])
    delta_t = np.sort(delta_t)  #sorted list of all timestamps that belong to node i
    if len(delta_t[0,:])>1: #must be at least 2 values
        for j in range(len(delta_t[0,:])-1):
            diff = delta_t[0,j+1]-delta_t[0,j] #time difference
            dT = np.append(dT,diff) #one gigantic list of timedifferences
            
plt.hist(dT,100)
plt.xlabel('inter arrival time')
plt.ylabel('frequency')
plt.show()
#%% 9
plt.close("all")

ExpVal = np.sum(Infections, axis = 1)/Nnodes
StandardDev = np.std(Infections, axis = 1)

t=np.linspace(1,tmax,len(ExpVal))
plt.axes(xlim=(1,tmax))
plt.xlabel('Time(s)')
plt.ylabel('Average Infected Nodes')
plt.title('Average Infected Nodes Versus Time With Corresponding Standard Deviation (G2)')
plt.errorbar(t,ExpVal,yerr = StandardDev, errorevery = 100, ecolor = 'r', color = 'k')
#%% 10
R = np.ones(Nnodes)*float('nan')

for k in range(Nnodes):
    for i in range(tmax):
       if Infections[i,k] > 0.8*Nnodes:
           R[k] = i+1 
           break
#%%11
plt.close("all")
C = g.transitivity_local_undirected(vertices = None, mode = "zero")
C_node = np.argsort(C)[::-1]+1
C.sort(reverse=True)
D = g.outdegree()
D_node = np.argsort(D)[::-1]+1
D.sort(reverse=True)
R_node = np.argsort(R)+1
f = np.linspace(0.05,0.5,10)
rd_f = np.zeros(10)
rc_f = np.zeros(10)
for i in range(10):
    size_R_f =int(round(f[i]*Nnodes))
    R_f = R_node[0:size_R_f]
    D_f = D_node[0:size_R_f]
    C_f = C_node[0:size_R_f]
    rd_f[i] = len(set(R_f).intersection(D_f))/size_R_f
    rc_f[i] = len(set(R_f).intersection(C_f))/size_R_f
plt.figure()
plt.axes(ylim=(0,1))
plt.xlabel('Fraction of top most influential nodes')
plt.ylabel('Recognition rate')
plt.title('Recognition rate using the degree of the nodes (G2)')
plt.plot(f,rd_f)

plt.figure()
plt.axes(ylim=(0,1))
plt.xlabel('Fraction of top most influential nodes')
plt.ylabel('Recognition rate')
plt.title('Recognition rate using the clustering coefficient of the nodes (G2)')
plt.plot(f,rc_f)
#%% 12
plt.close("all")
closeness = g.closeness()
closeness_node = np.argsort(closeness)[::-1]+1

rclose_f = np.zeros(10)
for i in range(10):
    size_R_f =int(round(f[i]*Nnodes))
    R_f = R_node[0:size_R_f]
    Close_f = closeness_node[0:size_R_f]
    rclose_f[i] = len(set(R_f).intersection(Close_f))/size_R_f
plt.figure()
plt.axes(ylim=(0,1))
plt.xlabel('Fraction of top most influential nodes')
plt.ylabel('Recognition rate')
plt.title('Recognition rate using the closeness of the nodes (G2)')
plt.plot(f,rclose_f)

"""MISSING SECOND METHOD"""
#%% Time node is reached
plt.close("all")
events = G.sort_values(by=['node1', 'node2'])

col1 = events['node1']
col2 = events['node2']
#col3 = data['timestamp']

col1 = col1.tolist()
col2 = col2.tolist()
#col3 = col3.tolist()
col=np.append(col1,col2)
countevents = pd.value_counts(col)
events_node=countevents.index
events_node=events_node.tolist()


revents_f = np.zeros(10)
for i in range(10):
    size_R_f =int(round(f[i]*Nnodes))
    R_f = R_node[0:size_R_f]
    events_f = events_node[0:size_R_f]
    revents_f[i] = len(set(R_f).intersection(events_f))/size_R_f
plt.figure()
plt.axes(ylim=(0,1))
plt.xlabel('Fraction of top most influential nodes')
plt.ylabel('Recognition rate')
plt.title('Recognition rate using the time of events of the nodes (G2)')
plt.plot(f,revents_f)
#%% 13
plt.close("all")

InfectionPerTimestep = np.zeros((tmax,Nnodes))
infections2 = np.zeros((tmax+1,Nnodes))
infections2[1:,:] = Infections
infections2 = np.delete(infections2,tmax,axis=0)
TimeInfectionPT= np.zeros((tmax,Nnodes))
R_accent = np.zeros(Nnodes)
R[Nnodes-1] = tmax

for j in range(Nnodes):
    if np.isnan(R[j]) == True:
        R[j] = tmax
    for i in range(int(R[j])):
        InfectionPerTimestep[i,j] = Infections[i,j] - infections2[i,j]
        TimeInfectionPT[i,j] = InfectionPerTimestep[i,j]*(i+1) 
    R_accent[j] = np.sum(TimeInfectionPT[:,j])/(0.8*Nnodes)

R_accent_node = np.argsort(R_accent)+1
f = np.linspace(0.05,0.5,10)
rd2_f = np.zeros(10)
rc2_f = np.zeros(10)
rr_f = np.zeros(10)


for i in range(10):
    size_R_accent_f =int(round(f[i]*Nnodes))
    R_accent_f = R_accent_node[0:size_R_accent_f]
    R_f = R_node[0:size_R_accent_f]
    D_f = D_node[0:size_R_accent_f]
    C_f = C_node[0:size_R_accent_f]
    rd2_f[i] = len(set(R_accent_f).intersection(D_f))/size_R_accent_f
    rc2_f[i] = len(set(R_accent_f).intersection(C_f))/size_R_accent_f
    rr_f[i] = len(set(R_accent_f).intersection(R_f))/size_R_accent_f
plt.figure()
plt.axes(ylim=(0,1))
plt.xlabel('Fraction of top most influential nodes')
plt.ylabel('Recognition rate')
plt.title('Recognition rate using the degree of the nodes (G2)')
plt.plot(f,rd2_f)

plt.figure()
plt.axes(ylim=(0,1))
plt.xlabel('Fraction of top most influential nodes')
plt.ylabel('Recognition rate')
plt.title('Recognition rate using the clustering coefficient of the nodes (G2)')
plt.plot(f,rc2_f)

plt.figure()
plt.axes(ylim=(0,1))
plt.xlabel('Fraction of top most influential nodes')
plt.ylabel('Recognition rate')
plt.title('Recognition rate using the 80 percent ranking R of the nodes (G2)')
plt.plot(f,rr_f)

#plt.figure()
#plt.plot(1/Average_Time_Infection)
#plt.figure()
#plt.plot(1/R)





