# -*- coding: utf-8 -*-
#Copy file for Cleo at 19_07, please delete
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy.random as rnd
import timeit
import random

#simulation = 'HS2011'
simulation = 'Haggle' 
#simulation = 'MIT'

if simulation == 'HS2011':
    data = pd.read_csv(r'thiers_2011.csv', delim_whitespace=True, header =None)
    data.columns = ['timestamp', 'node1', 'node2', 'triv1', 'triv2']
    data = data.drop(columns =['triv1','triv2'])
    data = data[['node1', 'node2', 'timestamp']]
    data['node2']+= 1000
    data['node1']+= 1000
    unique_node1 = len(data['node1'].unique())


    for i in range(len(data['node1'].unique())):
        data['node2'] = data['node2'].where(data['node2']!=sorted(data['node1'].unique())[i],other = i)
        data['node1'] = data['node1'].where(data['node1']!=sorted(data['node1'].unique())[i],other = i)     
    
    test = data['node2'].where(data['node2']>unique_node1)
    test = test.dropna()
    for i in range(len(test.unique())):
        data['node2'] = data['node2'].where(data['node2']!=sorted(test.unique())[i],other = i+unique_node1)


if simulation == 'HS2012':
    data = pd.read_csv(r'thiers_2012.csv', delim_whitespace=True, header =None)
    data.columns = ['timestamp', 'node1', 'node2', 'triv1', 'triv2']
    data = data.drop(columns =['triv1','triv2'])
    data = data[['node1', 'node2', 'timestamp']]
    unique_node1 = len(data['node1'].unique())

    for i in range(len(data['node1'].unique())):
        data['node2'] = data['node2'].where(data['node2']!=sorted(data['node1'].unique())[i],other = i)
        data['node1'] = data['node1'].where(data['node1']!=sorted(data['node1'].unique())[i],other = i)  
    
    test = data['node2'].where(data['node2']>unique_node1)
    test = test.dropna()
    for i in range(len(test.unique())):
        data['node2'] = data['node2'].where(data['node2']!=sorted(test.unique())[i],other = i+unique_node1)
 

if simulation == 'HS2013':
    data = pd.read_csv(r'HS2013.CSV', delim_whitespace=True, header =None)
    data.columns = ['timestamp', 'node1', 'node2', 'triv1', 'triv2']
    data = data.drop(columns =['triv1','triv2'])
    data = data[['node1', 'node2', 'timestamp']]
    data['node2']+= 1000
    data['node1']+= 1000
    unique_node1 = len(data['node1'].unique())
    
    for i in range(unique_node1):
        data['node2'] = data['node2'].where(data['node2']!=sorted(data['node1'].unique())[i],other = i)
        data['node1'] = data['node1'].where(data['node1']!=sorted(data['node1'].unique())[i],other = i)
    
    test = data['node2'].where(data['node2']>unique_node1)
    test = test.dropna()
    for i in range(len(test.unique())):
        data['node2'] = data['node2'].where(data['node2']!=sorted(test.unique())[i],other = i+unique_node1)    
      

if simulation == 'Haggle':
    data = pd.read_excel (r'data_Haggle_sorted.xlsx')
if simulation == 'MIT':
    data = pd.read_excel (r'MIT_data_sorted.xlsx')
    #data = pd.read_excel(r'C:\Users\cleoo\Documents\Complex Network\Complex_Networks\Final_assignment\MIT_data_sorted.xlsx')

nodes = data['node1'].drop_duplicates() + data['node2'].drop_duplicates()
nodes = nodes.drop_duplicates()

Nnodes = np.max([data['node1'].max(), data['node2'].max()])
Nlinks = len(data)
tmin = np.max([data['timestamp'].min()])
if simulation == 'HS2011':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20)
if simulation == 'HS2012':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20)
if simulation == 'HS2013':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20)
if simulation == 'Haggle':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20) 
if simulation == 'MIT':
    data['timestamp'] = ( data['timestamp'] - tmin)/600         #first timestamp is 0, every timestep is 10 minutes

tmax = int(data.timestamp.max())
percentage_dropped = 0.60 
#haggle (addapted is 0.006 and for MIT = 0.36)
#%% Analysis

"""Non temporal Situations are described and started here"""
plt.close("all")


start = timeit.default_timer()
tmax = int(data.timestamp.max())

if simulation == 'HS2011':
    gamma = 0
    beta = 1
    Tinf = 2821
if simulation == 'HS2013':
    gamma = 0
    beta = 1
    Tinf = 2821
if simulation == 'HS2012':
    gamma = 0
    beta = 1
    Tinf = 2821
if simulation == 'Haggle':
    gamma = 0
    beta = 1
    Tinf = 2821
if simulation == 'MIT':
    gamma = 0
    beta = 1
    Tinf = 1843
    
Tisolation = 5000

Infections = np.zeros([tmax,Nnodes])
#Removed = np.zeros([tmax,Nnodes, Nnodes])
Removed_total = np.zeros([tmax,Nnodes])
Susceptible = np.zeros([tmax,Nnodes])
data_dropped = data

"""Mittigation strategy"""
situations = ['No effects', 'Random', 'Isolation', 'Least used nodes', 'Max number of link']
choose_situation = situations[2]
T10 = 3000
Tbegin = T10
Tend = 40000.25*tmax


if choose_situation == 'Least used nodes':
    data_timeframe = data.loc[Tbegin <= data['timestamp']]
    data_timeframe = data.loc[data['timestamp'] <= Tend]
    duplicates = data_timeframe
    duplicates = data.pivot_table(index=['node1','node2'], aggfunc='size')
    duplicates = pd.Series.sort_values(duplicates,ascending=False)
    n_deleted_links = len(data_timeframe)*percentage_dropped

    som = 0
    for i in range(len(duplicates)):
        som = duplicates.values[-i] + som
        if som > n_deleted_links:
            row_stop = i
            som = som - duplicates.values[-i]
            print('Number of rows to delete:', i)
            break
    
    for i in range(row_stop):
        drop_indices = data_timeframe[(data_timeframe[['node1','node2']] == duplicates.index[-i]).all(1)].index.tolist()
        data_dropped = data_dropped.drop(drop_indices)
        if i  % 100 == 0:
            print('We are at:', round(i/row_stop*100), '%. Elapsed Time', round(timeit.default_timer()-start))
    
    n_removed = int(n_deleted_links - som)
    drop_indices = data[(data[['node1','node2']] == duplicates.index[-(row_stop+1)]).all(1)].index.tolist()
    data_dropped = data_dropped.drop(random.sample(drop_indices,n_removed))
    
    print(len(data)-len(data_dropped), 'links are deleted')
    

"""Starting from here is the evaluation of the infections"""


Aoud = np.eye(Nnodes) #starting nodes are infected
unit = np.eye(Nnodes)
isolated = np.zeros(Nnodes)
inf_t = np.zeros([Nnodes,2])

#!#! Copy this
Inf_time = np.eye(Nnodes)   #time node is infected
Removed = np.zeros([Nnodes, Nnodes])
#!#!

stop = timeit.default_timer()
print('Starting evaluation,elapsed time till now', round(stop-start))

for i in range(0,tmax):
    data_temp = data_dropped[data_dropped.timestamp==i].values
    A = np.zeros([Nnodes,Nnodes])
    w = int(len(data_temp))
    
    
    
    if data_temp.size:
        for j in range(w):
            p = 0#rnd.rand()
            if p<beta:    # When p is smaller than beta, (0.11<0.2) then the contact will be counted as an infection, if not, no infection so no changes in infection matrix
                A[int(data_temp[j,0]-1),int(data_temp[j,1]-1)] = 1
                A[int(data_temp[j,1]-1),int(data_temp[j,0]-1)] = 1
        Inf = np.dot(A+unit,Aoud) #infectable content
        Inf[Inf>0]=1
        
    #!#! copy from here
        
    if i > Tbegin: #start mitigation only in this window    
        Inf_time = Inf_time + Inf
        if choose_situation == 'Isolation': # isolation from t = i + Tisolation until infinity
            #isolated = np.argwhere(Inf_time>Tisolation) #list with indices of isolated nodes
            #Removed[isolated] = 1 #matrix with removed nodes at t=i, can be adjusted to have gamma
            #Removed_total[i,:] = np.sum(Removed,0) #total number of removed nodes per timestep per starting node
            #Inf[isolated]=0   #in isolation, you can't be infected again
            Inf = np.where(Inf_time>Tisolation, 0, Inf) #isolated nodes cannot be infective
            Removed = np.where(Inf_time>Tisolation, 1, Removed)
            Removed_total[i,:] = np.sum(Removed,0) #total number of removed nodes per timestep per starting node
    
    Aoud = Inf #- Removed[i,:,:] #current infected nodes
    Aoud[Aoud<0]=0
    
    #!#!
     
    #Removed_total[i,:] = np.sum(Removed[i,:,:], axis=0)      
    Infections[i,:] = np.sum(Aoud, axis=0)
    Susceptible[i,:] = Nnodes - Removed_total[i,:] - Infections[i,:]
    if i  % 1000 == 0:
        print('We are at:', round(i/tmax*100), '%')
    
stop = timeit.default_timer()
print('Elapsed Time:',stop-start)


'''Plotting'''

'''Observables'''
maxInf = np.max(np.sum(Infections,axis=1))/(Nnodes**2)*100
print('The average maximum number over infections is:',maxInf, '%')
tmaxinf = np.argmax(np.sum(Infections,axis=1))
print('The timestamp of the maximum number of infections is:',tmaxinf)
Scsleft = np.sum(Susceptible[-1])/(Nnodes**2)*100
print('There are',Scsleft,'% susceptible nodes left on average')

#%%
ExpVal = np.sum(Infections, axis = 1)/(Nnodes**2)*100  #percentage of total nodes
StandardDev = np.std(Infections, axis = 1)/Nnodes *100       # percentage of total nodes
ExpVal_rem = np.sum(Removed_total, axis = 1)/(Nnodes**2)*100  
StandardDev_rem = np.std(Removed_total, axis = 1)/Nnodes * 100 
ExpVal_sus = np.sum(Susceptible, axis = 1) /(Nnodes**2)*100 

t=np.linspace(1,tmax,len(ExpVal))

y1 = ExpVal
y2 = 100 - ExpVal_rem

plt.figure()
plt.errorbar(t,ExpVal+ExpVal_rem,yerr = StandardDev+StandardDev_rem, errorevery = 400, ecolor = 'r', color = 'k') #in isolation, removed nodes are still infected
plt.errorbar(t,ExpVal_rem,yerr = StandardDev_rem, errorevery = 452, ecolor = 'y', color = 'b')
#plt.errorbar(t,ExpVal_sus,yerr = 0, errorevery = 452, ecolor = 'y', color = 'c')


plt.legend(('Infected' , 'Removed'))
plt.xlabel('Timestamp')
plt.ylabel('Average Percentage of Nodes')
#plt.title(r'Average % of infected and removed nodes versus time with $\sigma$')

