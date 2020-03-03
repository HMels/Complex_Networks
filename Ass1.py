import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt

#data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
data = pd.read_excel (r'C:\Users\cleoo\Documents\Complex Network\manufacturing_emails_temporal_network.xlsx')
#%% 1)
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
print(g)

ED = np.sum(g.degree())/Nnodes
print(ED)
Density = g.density()
print(Density)
Dvar = np.var(list(g.degree()))
print(Dvar)

#%% 2)
from scipy.optimize import curve_fit
def powerlaw(x, a, k):
    return a*x**(k)

bin_heights, bin_borders, _ = plt.hist(g.degree(),bins=np.max(g.degree()),weights=np.ones(Nnodes)/Nnodes, histtype = 'step', log=False, label='data')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
centers = np.take(bin_centers, np.nonzero(bin_heights))
centers = centers.reshape((65,))
heights = np.take(bin_heights, np.nonzero(bin_heights))
heights = heights.reshape((65,))
popt, _ = curve_fit(powerlaw, centers, heights, p0=[10**(-1), -3.])

x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
plt.plot(x_interval_for_fit, powerlaw(x_interval_for_fit, *popt), label='fit')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Log(Degree)')
plt.ylabel('Log(Pr[D=k])')
plt.legend()

#%% 3)

rhoD = g.assortativity_degree()

#%% 4) 6)
#Generate an ER random graph to investigate small world property
ER_graph = igraph.Graph()
ER_graph = ER_graph.Erdos_Renyi(Nnodes,m=Nlinks,directed=False, loops=False)

C = g.transitivity_undirected()
CER = ER_graph.transitivity_undirected()

#%% 5) 6)
H = g.shortest_paths()
EH = np.sum(np.sum(H,axis=1),axis=0)/((Nnodes-1)*Nnodes) #EH = APL

APL = g.average_path_length()
APL_ER = ER_graph.average_path_length() #compare with random graph
diam = np.max(H)

#%% 7)
#Eigadj = g.eigen_adjacency()
Adj = g.get_adjacency()
Adj = np.array([Adj.data])
Eig = np.linalg.eig(Adj)
MaxEig = np.max(Eig[0])

#%% 8)
Lap = g.laplacian()
Lap = np.array([Lap])
Eiglap = np.linalg.eig(Lap) #stores eigenvalues and eigenvectors
Alg_con = np.sort(Eiglap[0])
Alg_con = Alg_con[:,1] #take second smallest eigenvalue