import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN

url= 'RF-LSTM-CEEMDAN/Dataset/UnivDorm_Prince.csv'
data = pd.read_csv(url)

new_data= data[(data['timestamp'] > '2015-03-01') & (data['timestamp'] < '2015-06-01')].interpolate(method='linear', limit_direction='backward')
dfs=new_data['Humidity'].values

emd = CEEMDAN(epsilon=0.05)
emd.noise_seed(12345)
IMFs = emd(dfs)
IMFs = IMFs.T

print(IMFs.shape)

fig, axs = plt.subplots(9)
#fig.suptitle('Sharing both axes')
axs[0].plot(dfs)
axs[0].set_title('Meteorology')
axs[1].plot(IMFs[:,0])
axs[1].set_title('IMF1')
axs[2].plot(IMFs[:,1])
axs[2].set_title('IMF2')
axs[3].plot(IMFs[:,2])
axs[3].set_title('IMF3')
axs[4].plot(IMFs[:,3])
axs[4].set_title('IMF4')
axs[5].plot(IMFs[:,4])
axs[5].set_title('IMF5')
axs[6].plot(IMFs[:,5])
axs[6].set_title('IMF6')
axs[7].plot(IMFs[:,6])
axs[7].set_title('IMF7')
axs[8].plot(IMFs[:,7])
axs[8].set_title('IMF8')
# axs[9].plot(IMFs[:,8])
# axs[9].set_title('IMF9')
# axs[10].plot(IMFs[:,9])
# axs[10].set_title('IMF10')
plt.show()


print(IMFs.shape)
full_imf = pd.DataFrame(IMFs)
data_imf = full_imf.T


plt.plot(new_data['energy'].values[200:700], color='black', linewidth=1, label='Energy')
plt.plot(new_data['TemperatureC'].values[200:700], color='orange', linewidth=1, label='Temperature')
plt.plot(new_data['Humidity'].values[200:700], color='blue', linewidth=1, label='Humidity')
plt.legend(loc='upper right')
# plt.ylabel('Energy (kW/h)')
plt.grid(axis='y')
plt.show()

url1= 'https://raw.githubusercontent.com/irenekarijadi/RF-LSTM-CEEMDAN/main/Dataset/data%20of%20UnivLab_Christy.csv'
data1 = pd.read_csv(url1)

new_data1= data1[(data1['timestamp'] > '2015-03-01') & (data1['timestamp'] < '2015-06-01')]
dfs1=new_data1['energy']

url2= 'https://raw.githubusercontent.com/irenekarijadi/RF-LSTM-CEEMDAN/main/Dataset/data%20of%20UnivClass_Abby.csv'
data2 = pd.read_csv(url2)

new_data2= data2[(data2['timestamp'] > '2015-03-01') & (data2['timestamp'] < '2015-06-01')]
dfs2=new_data2['energy']


url3= 'https://raw.githubusercontent.com/irenekarijadi/RF-LSTM-CEEMDAN/main/Dataset/data%20of%20Office_Abigail.csv'
data3 = pd.read_csv(url3)

new_data3= data3[(data3['timestamp'] > '2015-03-01') & (data3['timestamp'] < '2015-06-01')]
dfs3=new_data3['energy']


url4= 'https://raw.githubusercontent.com/irenekarijadi/RF-LSTM-CEEMDAN/main/Dataset/data%20of%20PrimClass_Jaden.csv'
data4 = pd.read_csv(url4)

new_data4= data4[(data4['timestamp'] > '2015-03-01') & (data4['timestamp'] < '2015-06-01')]
dfs4=new_data4['energy']

print(dfs.shape)
dfs=dfs[:2149]
dfs1=dfs1[:2149]
dfs2=dfs2[:2149]
dfs3=dfs3[:2149]
dfs4=dfs4[:2149]


font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11.,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 11,
}


'''
weeks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
xlabel = [i for i in range(0, 168, 24)]

plt.subplot(1,2,1)
plt.plot(dfs.values[24:192], color='black', linewidth=1, label='1st week')
plt.plot(dfs.values[192:360], color='orange', linewidth=1, label='2nd week')
plt.legend(loc='upper right', prop=font1)
plt.xticks(xlabel, [weeks[i//24] for i in xlabel])
plt.ylabel('Energy (kW/h)', font2)
plt.grid(axis='y')

plt.subplot(1,2,2)
plt.plot(dfs1.values[24:192], color='black', linewidth=1, label='1st week')
plt.plot(dfs1.values[192:360], color='orange', linewidth=1, label='2nd week')
plt.legend(loc='upper right', prop=font1)
plt.xticks(xlabel, [weeks[i//24] for i in xlabel])
plt.ylabel('Energy (kW/h)', font2)
plt.grid(axis='y')

plt.show()
'''




x=range(0,2149)
fig, ax = plt.subplots(5, 1,sharex=True,
                       figsize=(8, 6),tight_layout=True)
#fig.suptitle('Sharing both axes')
ax[0].plot(x, dfs)
ax[0].set_title('UnivDorm_Prince')
ax[1].plot(x, dfs1)
ax[1].set_title('UnivClass_Abby')
ax[2].plot(x, dfs2)
ax[2].set_title('UnivLab_Christy')
ax[3].plot(x, dfs3)
ax[3].set_title('Office_Abigail')
ax[4].plot(x, dfs4)
ax[4].set_title('PrimClass_Jaden')

fig.text(0.5, 0.0001, 'Time', ha='center')
fig.text(0.0001, 0.5, 'Hourly energy consumption (kW/h)', va='center', rotation='vertical')
plt.show()