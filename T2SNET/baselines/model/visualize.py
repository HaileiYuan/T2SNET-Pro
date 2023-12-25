import matplotlib.pyplot as plt
import numpy as np

#proposed method
proposed_method_univdorm = (np.load('Dor.npz')['prediction'],np.load('Dor.npz')['truth'])
proposed_method_univlab = (np.load('Lab.npz')['prediction'], np.load('Lab.npz')['truth'])
proposed_method_univclass = (np.load('UniClass.npz')['prediction'], np.load('UniClass.npz')['truth'])
proposed_method_office = (np.load('Office.npz')['prediction'], np.load('Office.npz')['truth'])
proposed_method_primclass = (np.load('PrimClass.npz')['prediction'], np.load('PrimClass.npz')['truth'])

plt.plot(proposed_method_univdorm[1], label="actual")
plt.plot(proposed_method_univdorm[0], label="predictions", linestyle='dashed')
plt.title('University Dormitory Building')
plt.legend()
plt.xlabel('Time')
plt.ylabel('kW')
plt.show()

plt.plot(proposed_method_univlab[1], label="actual")
plt.plot(proposed_method_univlab[0], label="predictions", linestyle='dashed')
plt.title('University Laboratory Building')
plt.legend()
plt.xlabel('Time')
plt.ylabel('kW')
plt.show()

plt.plot(proposed_method_univclass[1], label="actual")
plt.plot(proposed_method_univclass[0], label="predictions", linestyle='dashed')
plt.title('University Classroom Building')
plt.legend()
plt.xlabel('Time')
plt.ylabel('kW')
plt.show()


plt.plot(proposed_method_office[1], label="actual")
plt.plot(proposed_method_office[0], label="predictions", linestyle='dashed')
plt.title('Office Building')
plt.legend()
plt.xlabel('Time')
plt.ylabel('kW')
plt.show()



plt.plot(proposed_method_primclass[1], label="actual")
plt.plot(proposed_method_primclass[0], label="predictions", linestyle='dashed')
plt.title('Primary Classroom Building')
plt.legend()
plt.xlabel('Time')
plt.ylabel('kW')
plt.show()