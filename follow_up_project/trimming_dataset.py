# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:23:33 2020

@author: Mels
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel (r'data_Haggle_sorted.xlsx')

nodes = data['node1'].drop_duplicates() + data['node2'].drop_duplicates()
nodes = nodes.drop_duplicates()

Nnodes = np.max([data['node1'].max(), data['node2'].max()])
Nlinks = len(data)
tmin = np.max([data['timestamp'].min()])

data['timestamp'] = np.round(( data['timestamp'] - tmin)/20) 
Tinf = 1843

tmax = int(data.timestamp.max())

#%% removing empty timestamps
Nt = data.shape[0]
data_temp = data.values

j = 1
for i in range(1,Nt):    
    #if data_temp.timestamp[i] != data_temp.timestamp[i-1]:
     #   data_temp.timestamp[i] = j
      #  j += 1 
    if data_temp[i,3] != data_temp[i-1,3]:
        data_temp[i,3] = j
        j += 1 
        
dict = {'node1': data_temp[:,0], 'node2': data_temp[:,1],'weight': data_temp[:,2], 'timestamp': data_temp[:,3]}
data_temp = pd.DataFrame(dict)

data_temp.to_excel('data_Haggle_sorted_trimmed.xlsx')