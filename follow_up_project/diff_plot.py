import numpy as np
import matplotlib.pyplot as plt

HS11 = np.load(r'juiste aantal links gedropt/HS2011.npy')
HS11_iso = np.load(r'juiste aantal links gedropt/HS2011_isolation.npy')
HS11_LU = np.load(r'juiste aantal links gedropt/HS2011_leastused.npy')
HS12 = np.load(r'juiste aantal links gedropt/HS2012.npy')
HS12_iso = np.load(r'juiste aantal links gedropt/HS2012_isolation.npy')
HS12_LU = np.load(r'juiste aantal links gedropt/HS2012_leastused.npy')
HS13 = np.load(r'juiste aantal links gedropt/HS2013.npy')
HS13_iso = np.load(r'juiste aantal links gedropt/HS2013_isolation.npy')
HS13_LU = np.load(r'juiste aantal links gedropt/HS2013_leastused.npy')
MIT = np.load(r'juiste aantal links gedropt/MIT.npy')
MIT_iso = np.load(r'juiste aantal links gedropt/MIT_isolation.npy')
MIT_LU = np.load(r'juiste aantal links gedropt/MIT_leastused.npy')

HS11_diff = HS11 - HS11_iso
HS12_diff = HS12 - HS12_iso
HS13_diff = HS13 - HS13_iso
MIT_diff = MIT - MIT_iso
HS11_diff_LU = HS11 - HS11_LU
HS12_diff_LU = HS12 - HS12_LU
HS13_diff_LU = HS13 - HS13_LU
MIT_diff_LU = MIT - MIT_LU


HS11_cumsum = np.cumsum(HS11_diff)
HS12_cumsum = np.cumsum(HS12_diff)
HS13_cumsum = np.cumsum(HS13_diff)
MIT_cumsum = np.cumsum(MIT_diff)
HS11_cumsum_LU = np.cumsum(HS11_diff_LU)
HS12_cumsum_LU = np.cumsum(HS12_diff_LU)
HS13_cumsum_LU = np.cumsum(HS13_diff_LU)
MIT_cumsum_LU = np.cumsum(MIT_diff_LU)

plt.figure()
plt.plot(np.linspace(0,1,len(HS11_diff)), HS11_diff, label='HS11')
plt.plot(np.linspace(0,1,len(HS12_diff)), HS12_diff, label='HS12')
plt.plot(np.linspace(0,1,len(HS13_diff)), HS13_diff, label='HS13')
plt.plot(np.linspace(0,1,len(MIT_diff)), MIT_diff, label='MIT')
plt.plot(np.linspace(0,1,len(HS11_diff_LU)), HS11_diff_LU, label='HS11_LU')
plt.plot(np.linspace(0,1,len(HS12_diff_LU)), HS12_diff_LU, label='HS12_LU')
plt.plot(np.linspace(0,1,len(HS13_diff_LU)), HS13_diff_LU, label='HS13_LU')
plt.plot(np.linspace(0,1,len(MIT_diff_LU)), MIT_diff_LU, label='MIT_LU')

plt.xlabel('Time [%]')
plt.ylabel('diff Infected [%]')
plt.title('10% iso, No Recovery')
plt.legend()

plt.figure()
plt.plot(np.linspace(0,1,len(HS11_diff)), HS11_cumsum/len(HS11_diff), label='HS11')
plt.plot(np.linspace(0,1,len(HS12_diff)), HS12_cumsum/len(HS12_diff), label='HS12')
plt.plot(np.linspace(0,1,len(HS13_diff)), HS13_cumsum/len(HS13_diff), label='HS13')
plt.plot(np.linspace(0,1,len(MIT_diff)), MIT_cumsum/len(MIT_diff), label='MIT')
plt.plot(np.linspace(0,1,len(HS11_diff_LU)), HS11_cumsum_LU/len(HS11_diff_LU), label='HS11_LU')
plt.plot(np.linspace(0,1,len(HS12_diff_LU)), HS12_cumsum_LU/len(HS12_diff_LU), label='HS12_LU')
plt.plot(np.linspace(0,1,len(HS13_diff_LU)), HS13_cumsum_LU/len(HS13_diff_LU), label='HS13_LU')
plt.plot(np.linspace(0,1,len(MIT_diff_LU)), MIT_cumsum_LU/len(MIT_diff_LU), label='MIT_LU')

plt.xlabel('Time [%]')
plt.ylabel('cum diff Infected [%]')
plt.title('10% iso, No Recovery')
plt.legend()