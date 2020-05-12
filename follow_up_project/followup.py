# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import numpy.random as rnd
import timeit
"""
Created on Sun May 10 10:43:36 2020

@author: cleoo
"""

#simulation = 'Haggle' 
simulation = 'MIT'

if simulation == 'MIT':
    #B = pd.read_excel (r'MIT_data_sorted.xlsx')
    B = pd.read_excel(r'C:\Users\cleoo\Documents\Complex Network\Complex_Networks\Final_assignment\MIT_data_sorted.xlsx')
if simulation == 'Haggle':
    B = pd.read_excel (r'C:\Users\cleoo\Documents\Complex Network\Complex_Networks\Final_assignment\data_Haggle_sorted.xlsx')

#data = data.drop_duplicates()

data = B
Nnodes = np.max([data['node1'].max(), data['node2'].max()])
Nlinks = len(data)
tmin = np.max([data['timestamp'].min()])
if simulation == 'Haggle':
    data['timestamp'] = np.round(( data['timestamp'] - tmin)/20) 
if simulation == 'MIT':
    data['timestamp'] = ( data['timestamp'] - tmin)/600         #first timestamp is 0, every timestep is 10 minutes

tmax = int(data.timestamp.max())
n_links_t_nodes = Nlinks/tmax/Nnodes
#haggle (addapted is 0.006 and for MIT = 0.36)
#%% Analysis

"""Non temporal Situations are described and started here"""
plt.close("all")


start = timeit.default_timer()
tmax = int(data.timestamp.max())

if simulation == 'Haggle':
    gamma = 0
    beta = 1
    Tinf = 2821
if simulation == 'MIT':
    gamma = 0#0.00025
    beta = 1#0.009
    Tinf = 1843
Infections = np.zeros([tmax,Nnodes])
Removed = np.zeros([tmax,Nnodes, Nnodes])
Removed_total = np.zeros([tmax,Nnodes])
Susceptible = np.zeros([tmax,Nnodes])
data_dropped = data


"""Starting from here is the evaluation of the infections"""


Aoud = np.eye(Nnodes)
unit = np.eye(Nnodes)
isolated = np.zeros(Nnodes)
inf_t = np.zeros([Nnodes,2])


stop = timeit.default_timer()
print('Starting evaluation,elapsed time till now', round(stop-start))

for i in range(0,tmax):
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
        Aoud = Inf - Removed[i,:,:] #current infected nodes
        Aoud[Aoud<0]=0
    if False: #removal of nodes if true
        Removed[(i+Tinf),:,:] = Aoud
        Removed[i+1,:,:] = Removed[i+1,:,:]+Removed[i,:,:]
        Removed[i+1,:,:][Removed[i+1,:,:]>1] = 1
        
        locate = np.argwhere(Aoud>0) #search for infected nodes
        for l in locate: #makes sure removed and infected dont overlap
            if Removed[i,l[0],l[1]] == 1:
                Aoud[l[0],l[1]] = 0
      
    Removed_total[i,:] = np.sum(Removed[i,:,:], axis=0)      
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
ExpVal = np.sum(Infections, axis = 1)/(Nnodes**2)*100       #percentage of total nodes
StandardDev = np.std(Infections, axis = 1)/Nnodes *100       # percentage of total nodes
ExpVal_rem = np.sum(Removed_total, axis = 1)/(Nnodes**2)*100  
StandardDev_rem = np.std(Removed_total, axis = 1)/Nnodes * 100 
ExpVal_sus = np.sum(Susceptible, axis = 1)/(Nnodes**2)*100

T10 = min(np.argwhere(ExpVal>=10))
T20 = min(np.argwhere(ExpVal>=20))
T40 = min(np.argwhere(ExpVal>=40))
T60 = min(np.argwhere(ExpVal>=60))
T80 = 0#min(np.argwhere(ExpVal>=80))
T90 = 0#min(np.argwhere(ExpVal>=90))
timestamps_inf = np.array([T10,T20,T40,T60,T80,T90])
#%%
ExpVal = np.sum(Infections, axis = 1)/(Nnodes**2)*100       #percentage of total nodes
StandardDev = np.std(Infections, axis = 1)/Nnodes *100       # percentage of total nodes
ExpVal_rem = np.sum(Removed_total, axis = 1)/(Nnodes**2)*100  
StandardDev_rem = np.std(Removed_total, axis = 1)/Nnodes * 100 
ExpVal_sus = np.sum(Susceptible, axis = 1)/(Nnodes**2)*100 

t=np.linspace(1,tmax,len(ExpVal))

y1 = ExpVal
y2 = 100 - ExpVal_rem
if simulation == 'Haggle':
    x=t
if simulation == 'MIT':
    x=t#t/6/24 

plt.figure()
#plt.axes(ylim=(0,100),xlim=(0,np.max(x)))
if simulation == 'Haggle':
    plt.errorbar(x,ExpVal,yerr = StandardDev, errorevery = 4000, ecolor = 'r', color = 'k')
    plt.errorbar(x,100 - ExpVal_rem,yerr = StandardDev_rem, errorevery = 4520, ecolor = 'y', color = 'b')
else:
    plt.errorbar(x[0:300],ExpVal[0:300],yerr = StandardDev[0:300], errorevery = 400, ecolor = 'r', color = 'k')
    plt.errorbar(x[0:300],100 - ExpVal_rem[0:300],yerr = StandardDev_rem[0:300], errorevery = 452, ecolor = 'y', color = 'b')

plt.legend(('Infected' , 'Removed'))
plt.xlabel('Timestamp')
plt.ylabel('Average Percentage of Nodes')
#plt.title(r'Average % of infected and removed nodes versus time with $\sigma$')

plt.figure()
plt.plot(x,y1,'k',x,y2,'k')
plt.axes(ylim=(0,100),xlim=(0,np.max(x)))
plt.fill_between(x,y1,y2,where=y2>=y1,facecolor = 'teal')
plt.fill_between(x,0,y1,facecolor = 'r')
plt.fill_between(x,y2,100,facecolor = 'silver')
#plt.title('Average % of infected and removed nodes versus time')
plt.xlabel('Timestamp')
plt.ylabel('Average Percentage of Nodes')
plt.legend(('Susceptible','Infected' , 'Removed'))

plt.figure()
plt.axes(ylim=(0,100),xlim=(0,np.max(x)))
plt.plot(x,ExpVal, 'k')
plt.plot(x,ExpVal_sus, 'b')
plt.plot(x,ExpVal_rem, 'r')
#plt.title('SIR Model')
plt.legend(('Infected' ,'Susceptible', 'Removed'))
plt.xlabel('Timestamp')
plt.ylabel('Average Percentage of Nodes')