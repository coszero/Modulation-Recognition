import numpy as np
import matplotlib.pyplot as plt 

curve_F3 = np.loadtxt('./result/CUM_NN_F3_L100_50000.txt', delimiter = ',', dtype = float)
curve_F9 = np.loadtxt('./result/CUM_NN_F9_L100_50000.txt', delimiter = ',', dtype = float)
curve_TC = np.loadtxt('./result/cum_based_all_L100.txt', delimiter = ',', dtype = float)
#plot the curve of SNR and Total Accuracy
plt.figure()
plt.plot(curve_F3[0,:],curve_F3[1,:],'bo-')
plt.plot(curve_F9[0,:],curve_F9[1,:],'rs-')
plt.plot(curve_TC[0,:],curve_TC[1,:],'y^-')
plt.legend(['FCNN based on 3 features','FCNN based on 9 features','TC Method'],loc=4)
plt.xlabel('SNR/dB')
plt.ylabel('Total Accuracy')
plt.title('Total accuracy and signal-to-noise ratio')
plt.show()

#plot the curve of SNR and Accuracy of each categories
NClass=4
output_label=np.array(['BPSK','QPSK','8PSK','16QAM'])
for i in range(NClass):
    plt.subplot(2,2,i+1);
    plt.plot(curve_F3[0,:],curve_F3[i+2,:],'bo-')
    plt.plot(curve_F9[0,:],curve_F9[i+2,:],'rs-')
    plt.legend(['3 features','9 features'],loc=4)
    plt.xlabel('SNR/dB')
    plt.ylabel(output_label[i]+' accuracy')
    plt.title(output_label[i])
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
