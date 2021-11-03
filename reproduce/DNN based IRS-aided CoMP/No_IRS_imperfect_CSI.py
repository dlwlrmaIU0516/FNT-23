import numpy as np
from parameters import p
from f_channel_gen import channel_realization
from torch_NN import construct_channel
import matplotlib.pyplot as plt
from alternaing_optimization_No_IRS import AO_cvx
import os
import warnings
import scipy.io as sio

warnings.filterwarnings(action='ignore')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

p['batch_size'] = 1
test_P_tx_dB = range(p['Tx_P_dB']-10,p['Tx_P_dB']+10,2)
test_P_tx = 10**(np.asarray(test_P_tx_dB)/10)
avg_min_rate = np.zeros(np.size(test_P_tx))
num_experiments = 10
idx = 0
for P_idx in test_P_tx:
    min_rate_temp = []
    for exp_idx in range(num_experiments):
        p['Tx_P'] = P_idx
        H = channel_realization(p)
        min_rate_temp = np.append(min_rate_temp,AO_cvx(p,H)[-1])
        if exp_idx % 10 == 0:
            print('P_tx [dB] : ',test_P_tx_dB[idx],', # experiments : ',exp_idx,',converge : ',min_rate_temp)
    avg_min_rate[idx] = np.mean(min_rate_temp)
    idx += 1



path = './fig/Tx_P[dB]_{}_M_{}_imperfect_gamma_{}'
try:
    if not(os.path.isdir(path.format(p['Tx_P_dB'],p['M'],p['gamma']))):
        os.makedirs(os.path.join(path.format(p['Tx_P_dB'],p['M'],p['gamma'])))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise

plt.figure(1)
plt.grid()
plt.plot(test_P_tx_dB,avg_min_rate)
plt.legend(['BCD'])
plt.xlabel('P_tx [dB]')
plt.ylabel('Avg. min rate [bps/Hz]')
#plt.show()

save_mat_template_avg_min_rate = './fig/Tx_P[dB]_{}_M_{}_imperfect_gamma_{}/No_IRS_avg_min_rate.mat'
save_fig_template_avg_min_rate = './fig/Tx_P[dB]_{}_M_{}_imperfect_gamma_{}/No_IRS_avg_min_rate.png'
sio.savemat(save_mat_template_avg_min_rate.format(p['Tx_P_dB'],p['M'],p['gamma']), {'avg_min_rate':avg_min_rate})
plt.savefig(save_fig_template_avg_min_rate.format(p['Tx_P_dB'],p['M'],p['gamma']))