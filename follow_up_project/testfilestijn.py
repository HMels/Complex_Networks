# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy.random as rnd
import timeit
import random

simulation = 'HS2011'
#simulation = 'Haggle' 
#simulation = 'MIT'
#simulation = 'Haggle_trimmed'  #don't forget to delete the first column in excel
                                #look at trimming_dataset.py to trim other datasets of 
                                #unnused timestamps


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
      
if simulation == 'Haggle_trimmed':
    data = pd.read_excel (r'data_Haggle_sorted_trimmed.xlsx')
    
if simulation == 'Haggle':
    data = pd.read_excel (r'data_Haggle_sorted.xlsx')

    
if simulation == 'MIT':
    data = pd.read_excel (r'MIT_data_sorted.xlsx')
    
nodes = data['node1'].drop_duplicates() + data['node2'].drop_duplicates()
nodes = nodes.drop_duplicates()

Nnodes = np.max([data['node1'].max(), data['node2'].max()])
Nlinks = len(data)
tmin = np.max([data['timestamp'].min()])
if simulation == 'HS2011':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20)
    Tinf = 1843
if simulation == 'HS2012':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20)
    Tinf = 1843
if simulation == 'HS2013':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20)
    Tinf = 1843
if simulation == 'Haggle':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20) 
    Tinf = 1843
if simulation == 'MIT':
    data['timestamp'] = ( data['timestamp'] - tmin)/600
    Tinf = 1843
if simulation == 'Haggle_trimmed':
    data['timestamp'] = ( data['timestamp'] - tmin)
    Tinf = 1843

tmax = int(data.timestamp.max())
gamma = 0
beta = 1

#%% Analysis
"""Non temporal Situations are described and started here"""
plt.close("all")

start = timeit.default_timer()

Infections = np.zeros([tmax,Nnodes])
#Removed = np.zeros([tmax,Nnodes, Nnodes])
Removed_total = np.zeros([tmax,Nnodes])
Susceptible = np.zeros([tmax,Nnodes])
data_dropped = data

if True: #if you want a fixed infection time
    Inf_time = np.eye(Nnodes)   #time node is infected
    Removed = np.zeros([Nnodes, Nnodes])

"""Mittigation strategy"""

situations = ['No effects', 'Random', 'Isolation', 'Least used links', 'Max number of links']
choose_situation = situations[3]

#T10 = 1000
#Tbegin = T10
mitigation_applied = False
begin_infection_percentage = int(0.1*Nnodes)

Tend = tmax
percentage_dropped = 0.1

Tisolation = 5000
    
"""Starting from here is the evaluation of the infections"""

Aoud = np.eye(Nnodes)
unit = np.eye(Nnodes)
isolated = np.zeros(Nnodes)
inf_t = np.zeros([Nnodes,2])
Tbegin = tmax #will be changed in the loop

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
    
    
    if i > Tbegin: #start mitigation only in this window    
        Inf_time = Inf_time + Inf
        if choose_situation == 'Isolation': # isolation from t = i + Tisolation until infinity
            Inf = np.where(Inf_time>Tisolation, 0, Inf) #isolated nodes cannot be infective
            Removed = np.where(Inf_time>Tisolation, 1, Removed) #isolated nodes are removed
            Removed_total[i,:] = np.sum(Removed,0) #total number of removed nodes per timestep per starting node

    Aoud = Inf #- Removed[i,:,:] #current infected nodes
    Aoud[Aoud<0]=0    
            
    Infections[i,:] = np.sum(Aoud, axis=0)
    Susceptible[i,:] = Nnodes - Removed_total[i,:] - Infections[i,:]
    
    if np.average(Infections[i,:]) > begin_infection_percentage and mitigation_applied == False:
        Tbegin = i
        mitigation_applied = True
        print('Start time =', Tbegin, 'Infected at this point:', np.average(Infections[i,:]))
        
        if choose_situation != 'No effects':
            print('Links are being deleted')

        if choose_situation == 'Least used links':
            data_timeframe = data.loc[Tbegin <= data['timestamp']]
            data_timeframe = data_timeframe.loc[data_timeframe['timestamp'] <= Tend]
            duplicates = data_timeframe.pivot_table(index=['node1','node2'], aggfunc='size')
            duplicates = pd.Series.sort_values(duplicates,ascending=False)
            n_deleted_links = len(data_timeframe)*percentage_dropped
            
            som = 0
            for i in range(1,len(duplicates)+1):
                som = duplicates.values[-i] + som
                if som > n_deleted_links:
                    row_stop = i-1
                    som = som - duplicates.values[-i]
                    print('Number of rows to delete:', i)
                    break
                
            for i in range(row_stop):
                drop_indices = data_timeframe[(data_timeframe[['node1','node2']] == duplicates.index[-i]).all(1)].index.tolist()
                data_dropped = data_dropped.drop(drop_indices)
                #if i  % 100 == 0:
                    # print('We are at:', round(i/row_stop*100), '%. Elapsed Time', round(timeit.default_timer()-start))
    
            n_removed = int(n_deleted_links - som)
            drop_indices = data[(data[['node1','node2']] == duplicates.index[-(row_stop+1)]).all(1)].index.tolist()
            data_dropped = data_dropped.drop(random.sample(drop_indices,n_removed))
            
            print(len(data)-len(data_dropped), 'links are deleted')
  
      
    if i  % 2000 == 0:
        print('We are at:', round(i/tmax*100), '%')
    
stop = timeit.default_timer()

'''Observables'''
maxInf = np.max(np.sum(Infections,axis=1))/(Nnodes**2)*100
print('The average maximum percentage of infections is:',maxInf, '%')
tmaxinf = np.argmax(np.sum(Infections,axis=1))
print('The timestamp of the maximum percentage of infections is:',tmaxinf)
Scsleft = np.sum(Susceptible[-1])/(Nnodes**2)*100
print('There are',Scsleft,'% susceptible nodes left on average')

#%%
ExpVal = np.sum(Infections, axis = 1)/(Nnodes**2)*100  #percentage of total nodes
StandardDev = np.std(Infections, axis = 1)/Nnodes *100       # percentage of total nodes
ExpVal_rem = np.sum(Removed_total, axis = 1)/(Nnodes**2)*100  
StandardDev_rem = np.std(Removed_total, axis = 1)/Nnodes*100 
ExpVal_sus = np.sum(Susceptible, axis = 1) /(Nnodes**2)*100 

t=np.linspace(1,tmax,len(ExpVal))

plt.figure()
if choose_situation == 'Isolation':
    #Isolated nodes (removed) should still be counted as infected
    plt.errorbar(t,ExpVal + ExpVal_rem,yerr = StandardDev + StandardDev_rem, errorevery = 400, ecolor = 'r', color = 'k')
else:
    plt.errorbar(t,ExpVal,yerr = StandardDev, errorevery = 400, ecolor = 'r', color = 'k')
plt.errorbar(t,ExpVal_rem,yerr = StandardDev_rem, errorevery = 452, ecolor = 'y', color = 'b')

plt.legend(('Infected' , 'Removed'))
plt.xlabel('Timestamp')
plt.ylabel('Average Percentage of Nodes')
#plt.title(r'Average % of infected and removed nodes versus time with $\sigma$')

#%% This script can be used to calculate at with time step 10% of the nodes is infected etc.
#Use this to compare time windows between datasets
ExpVal = np.sum(Infections, axis = 1)/(Nnodes**2)*100       #percentage of total nodes
StandardDev = np.std(Infections, axis = 1)/Nnodes *100       # percentage of total nodes
ExpVal_rem = np.sum(Removed_total, axis = 1)/(Nnodes**2)*100  
StandardDev_rem = np.std(Removed_total, axis = 1)/Nnodes * 100 
ExpVal_sus = np.sum(Susceptible, axis = 1)/(Nnodes**2)*100

T10 = min(np.argwhere(ExpVal>=10))
T20 = min(np.argwhere(ExpVal>=20))
T40 = min(np.argwhere(ExpVal>=40))
T60 = min(np.argwhere(ExpVal>=60))
T80 = 0 #min(np.argwhere(ExpVal>=80))
T90 = 0 #min(np.argwhere(ExpVal>=90))
timestamps_inf = np.array([T10,T20,T40,T60,T80,T90])