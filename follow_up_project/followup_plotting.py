import numpy as np
import matplotlib.pyplot as plt


#plot no mitigation, no recovery and transmission = 1
HS11 = np.load(r'HS2011.npy')
HS11_a = np.load(r'LU_10percent_dropped/HS2011_10_20.npy')
HS11_b = np.load(r'LU_10percent_dropped/HS2011_10_40.npy')
HS11_c = np.load(r'LU_10percent_dropped/HS2011_10_60.npy')
HS11_d = np.load(r'LU_10percent_dropped/HS2011_10_80.npy')
HS11_e = np.load(r'LU_10percent_dropped/HS2011_10_100.npy')


#HS12 = np.load(r'HS2012.npy')
#HS13 = np.load(r'HS2013.npy')
#Haggle = np.load(r'Haggle.npy')
#MIT = np.load(r'MIT.npy')
#MIT2 = np.load(r'MIT_10percent_LUN.npy')
#MIT3 = np.load(r'MIT_20percent_LUN.npy')


plt.figure()
plt.plot(np.linspace(0,1,len(HS11)), HS11, label='HS11')
plt.plot(np.linspace(0,1,len(HS11_a)), HS11_a, label='10-20%')
plt.plot(np.linspace(0,1,len(HS11_b)), HS11_b, label='10-40%')
plt.plot(np.linspace(0,1,len(HS11_c)), HS11_c, label='10-60%')
plt.plot(np.linspace(0,1,len(HS11_d)), HS11_d, label='10-80%')
plt.plot(np.linspace(0,1,len(HS11_e)), HS11_e, label='10-100%')


#plt.plot(np.linspace(0,1,len(HS12)), HS12, label='HS12')
#plt.plot(np.linspace(0,1,len(HS13)), HS13, label='HS13')
#plt.plot(np.linspace(0,1,len(Haggle)), Haggle, label='Haggle')
#plt.plot(np.linspace(0,1,len(MIT)), MIT, label='MIT')
#plt.plot(np.linspace(0,1,len(MIT2)), MIT2, label='10%')
#plt.plot(np.linspace(0,1,len(MIT3)), MIT3, label='20%')


plt.xlabel('Time [%]')
plt.ylabel('Infected [%]')
plt.title('Windowed LU, No Recovery')
plt.legend()