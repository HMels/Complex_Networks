import numpy as np
import matplotlib.pyplot as plt


#plot no mitigation, no recovery and transmission = 1
HS11 = np.load(r'HS2011.npy')
HS12 = np.load(r'HS2012.npy')
HS13 = np.load(r'HS2013.npy')
Haggle = np.load(r'Haggle.npy')
MIT = np.load(r'MIT.npy')

plt.figure()
plt.plot(np.linspace(0,1,len(HS11)), HS11, label='HS11')
plt.plot(np.linspace(0,1,len(HS12)), HS12, label='HS12')
plt.plot(np.linspace(0,1,len(HS13)), HS13, label='HS13')
plt.plot(np.linspace(0,1,len(Haggle)), Haggle, label='Haggle')
plt.plot(np.linspace(0,1,len(MIT)), MIT, label='MIT')
plt.xlabel('Time [%]')
plt.ylabel('Infected [%]')
plt.title('No Mitigation, No Recovery')
plt.legend()