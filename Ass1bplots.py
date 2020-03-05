import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import igraph
import scipy.stats as ss

Tdata = np.load(r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\num_infArr.npy')
data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
Nnodes = 167
perc80 = 134
perc80index = np.zeros(Nnodes)
C = np.zeros(Nnodes)
D = np.zeros(Nnodes)
#%% 9)
Nave = np.sum(Tdata,axis=1)/Nnodes
Nvar = np.var(Tdata,axis=1)
Nstd = np.sqrt(Nvar)
t = np.linspace(1,len(Nave),len(Nave))

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set(xlabel=r'timestamp', ylabel=r'Infected nodes')
plt.errorbar(t,Nave,yerr=Nstd,ecolor='r')
plt.title('Errorbar plot of average infected nodes on timestamp t')
plt.show()

#%% 10)
for i in range(Nnodes):
    perc80index[i] = np.argmax(Tdata[:,i]>=134)
    if perc80index[i] ==0:
        perc80index[i] = 'NaN'
rank_perc80 = ss.rankdata(perc80index,method='min')

#%% 11)
G = data.drop(['timestamp'],axis=1)
B = G.drop_duplicates()
Nnodes = np.max([data['node1'].max(), data['node2'].max()])
Nlinks = len(B)
g = igraph.Graph()
g.add_vertices(Nnodes)

col1 = B['node1']; col2 = B['node2']
col1 = col1.tolist(); col2 = col2.tolist()

for i in range(Nlinks):
    g.add_edges([(col1[i]-1,col2[i]-1)]) #nodes are names 0 to 166
for i in range(Nnodes):    
    C[i] = g.transitivity_local_undirected(i)
    D[i] = g.outdegree(i)
rankC = ss.rankdata(-C,method='min')
rankD = ss.rankdata(-D,method='min')

f = np.linspace(0.05,0.5,10)
for i in f:
    Rf =int(round(i*Nnodes))
    
