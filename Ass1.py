import numpy as np
import pandas as pd
import igraph
import matplotlib.pyplot as plt

data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
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

#hist = np.histogram(g.degree(),np.max(g.degree()))
plt.hist(g.degree(),bins=np.max(g.degree()),weights=np.ones(Nnodes)/Nnodes)
plt.xlabel('Degree')
plt.ylabel('Pr[D=k]')
plt.title('Degree distribution')
plt.show()

#%% 3)

rhoD = g.assortativity_degree()

#%% 4)

C = g.transitivity_undirected()

#%% 5)
H = g.shortest_paths()
EH = np.sum(np.sum(H,axis=1),axis=0)/((Nnodes-1)*Nnodes)
APL = g.average_path_length()
diam = np.max(H)

#%% 7)
#Eigadj = g.eigen_adjacency()
Adj = g.get_adjacency()
Adj = np.array([Adj.data])
Eig = np.linalg.eig(Adj)
MaxEig = np.max(Eig)

#%% 8)
Lap = g.laplacian()
Lap = np.array([Lap])
Eiglap = np.linalg.eig(Lap)
