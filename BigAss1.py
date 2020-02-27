import numpy as np
import pandas as pd
import igraph as igraph
import matplotlib.pyplot as plt

#data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
data = pd.read_excel (r'C:\Users\rixtb\Documents\Master\Data analysis\Datasets\manufacturing_emails_temporal_network.xlsx')
#data = pd.read_excel (r'C:\Users\rixtb\Documents\Master\Data analysis\Datasets\oefenset.xlsx')
#%%
Nnodes = np.max([data['node1'].max(), data['node2'].max()])
G = data.drop(['timestamp'],axis=1)
B = G.drop_duplicates()
g = igraph.Graph()
g.add_vertices(Nnodes)

Nlinks = len(B)

col1 = B['node1']; col2 = B['node2']
col1 = col1.tolist(); col2 = col2.tolist()

for i in range(Nlinks):
    g.add_edges([(col1[i]-1,col2[i]-1)])

ED = g.degree()
print(ED)
p = g.density(loops=False) #ratio between the actual links and the possible edges
print(p)
average_degree = 2* Nlinks/Nnodes #see slides lecture 1
print(average_degree)
var = np.var(ED)
print(var)

plt.hist(ED,bins=Nnodes)

plt.plot()

pd = g.assortativity_degree()
print(pd)
clust_coef = g.transitivity_undirected()
print(clust_coef)

average_path_length=g.average_path_length()
print(average_path_length)
hopcount_max = np.max(g.shortest_paths())
print(hopcount_max)
#A = (g.get_adjacency())
#print(A)
eigenvalueA= g.eigen_adjacency()
print(eigenvalueA)

