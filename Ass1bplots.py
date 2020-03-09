import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import igraph
import scipy.stats as ss

Tdata = np.load(r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\G3_numinf.npy')
data = pd.read_excel (r'C:\Users\Thierry\Documents\Studie\TU Delft Applied Physics\CS4195 Modeling and Data Analysis in Complex Networks\Assignment1\manufacturing_emails_temporal_network.xlsx')
Nnodes = 167
perc80 = 134
perc80index = np.zeros(Nnodes)
C = np.zeros(Nnodes)
D = np.zeros(Nnodes)
Close = np.zeros(Nnodes)
Btween = np.zeros(Nnodes)
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
rank_perc80ind = rank_perc80.argsort()

#%% 11) 12)
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
    Close[i] = g.closeness(i)
    Btween[i] = g.betweenness(i)
rankC = ss.rankdata(-C,method='min')
rankCind = rankC.argsort()
rankD = ss.rankdata(-D,method='min')
rankDind = rankD.argsort()
rankClose = ss.rankdata(-Close,method='min')
rankCloseind = rankClose.argsort()
rankBtween = ss.rankdata(-Btween,method='min')
rankBtweenind = rankBtween.argsort()

f = np.rint(np.linspace(0.05,0.5,10)*Nnodes)
RfDf = np.zeros(len(f))
RfCf = np.zeros(len(f))
RfClf = np.zeros(len(f))
RfBf = np.zeros(len(f))
k = 0
for i in f:
    Cf = np.zeros(int(i))
    Df = np.zeros(int(i))
    Clf = np.zeros(int(i))
    Bf = np.zeros(int(i))
    for j in range(int(i)):
        abs_Rf = i
        Rf = np.where(rank_perc80 ==1)
        Df[j] = rankDind[j]
        Cf[j] = rankCind[j]
        Clf[j] = rankCloseind[j]
        Bf[j] = rankBtweenind[j]
    RfDf[k] = len(list(set(Rf[0]).intersection(Df)))/abs_Rf
    RfCf[k] = len(list(set(Rf[0]).intersection(Cf)))/abs_Rf
    RfClf[k] = len(list(set(Rf[0]).intersection(Clf)))/abs_Rf
    RfBf[k] = len(list(set(Rf[0]).intersection(Bf)))/abs_Rf
    k = k+1

F = np.linspace(0.05,0.5,10)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set(xlabel='f',ylabel='recognition rate')
plt.plot(F,RfCf,F,RfDf)
plt.legend(('RfCf', 'RfDf'),
                   loc='upper left')
plt.title('Recognition rate of Degree and Clustering coefficient')

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set(xlabel='f',ylabel='recognition rate')
plt.plot(F,RfClf,F,RfBf)
plt.legend(('RfClf', 'RfBf'),
                   loc='lower left')
plt.title('Recognition rate of Closeness and Betweenness')

#%% 13)
T80 = np.zeros(Nnodes)
for j in range(Nnodes):
    perc80_point = False
    for i in range(1,len(Tdata)):
        if Tdata[i,j] < 134 and perc80_point == False:
            T80[j] = T80[j] + (Tdata[i,j]-Tdata[i-1,j])*i
        elif Tdata[i,j] >= 134 and perc80_point == False:
            T80[j] = T80[j]/57791
            perc80_point = True
        elif i == 57791 and perc80_point == False:
            Tdiff = 134-Tdata[i,j]
            (T80[j]+Tdiff)/i
            perc80_point = True
            
rank_T80 = ss.rankdata(-T80,method='min')
rank_T80ind = rank_T80.argsort()

R2fCf = np.zeros(len(f))
R2fDf = np.zeros(len(f))
R2fRf = np.zeros(len(f))
k = 0
for i in f:
    Cf = np.zeros(int(i))
    Df = np.zeros(int(i))
    Rf = np.zeros(int(i))
    R2f = np.zeros(int(i))
    for j in range(int(i)):
        abs_Rf = i
        R2f[j] = rank_T80ind[j]
        Rf[j] = rank_perc80ind[j]
        Df[j] = rankDind[j]
        Cf[j] = rankCind[j]
    R2fDf[k] = len(list(set(R2f).intersection(Df)))/abs_Rf
    R2fCf[k] = len(list(set(R2f).intersection(Cf)))/abs_Rf
    R2fRf[k] = len(list(set(R2f).intersection(Rf)))/abs_Rf
    k = k+1
    
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.set(xlabel='f',ylabel='recognition rate')
plt.plot(F,R2fCf,F,R2fDf,F,R2fRf)
plt.legend(('R_primefCf', 'R_primefDf','R_primefRf'),
                   loc='upper left')
plt.title('Recognition rate of Degree, Clustering coefficient and R')