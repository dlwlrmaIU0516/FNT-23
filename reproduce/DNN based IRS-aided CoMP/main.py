import numpy as np
from parameters import p
from f_channel_gen import channel_realization
from torch_NN import IRS_CoMP_Net,construct_channel,loss_calculator,train
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import os
import warnings
import scipy.io as sio

warnings.filterwarnings(action='ignore')
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")
print('cuda index:', torch.cuda.current_device())
print('number of gpu:', torch.cuda.device_count())
print('graphic name:', torch.cuda.get_device_name())
cuda = torch.device('cuda')
print(cuda)

# GPU 할당 변경하기
GPU_NUM = 0 # 원하는 GPU 번호 입력
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check

# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')

    
NN_CoMP = IRS_CoMP_Net(p['N_t'],p['N_r'],p['M'],p['num_BS'],p['K'],p['d'],p['batch_size'],p['Tx_P'])
optimizer = optim.SGD(NN_CoMP.parameters(), lr=p['lr'])
optimizer
NN_CoMP.to(device)
object = []
for idx in range(p['iter']):
    H = channel_realization(p)
    BS_UE, BS_IRS, IRS_UE = construct_channel(H,p)
    BS_precoder, ISR_reflecter = NN_CoMP(BS_UE.to(device),BS_IRS.to(device),IRS_UE.to(device))
    avg_min_rate = loss_calculator(H,p,BS_precoder.to('cpu'), ISR_reflecter.to('cpu'))
    loss = -avg_min_rate
    train(loss,optimizer)
    temp = avg_min_rate.detach().numpy()
    object = np.append(object,temp)
    if idx % 10 == 0:
        print('Iter ',idx,': ',temp)




path = './fig/Tx_P[dB]_{}_M_{}'
try:
    if not(os.path.isdir(path.format(p['Tx_P_dB'],p['M']))):
        os.makedirs(os.path.join(path.format(p['Tx_P_dB'],p['M'])))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise


plt.figure(1)
plt.grid()
plt.semilogy(range(p['iter']),object)
plt.legend(['Proposed (P_tx = 10 [dB])'])
plt.xlabel('Iteration')
plt.ylabel('Avg. min rate [bps/Hz]')
# plt.show()
save_mat_template_training_converge = './fig/Tx_P[dB]_{}_M_{}/DL_converge.mat'
save_fig_template_training_converge = './fig/Tx_P[dB]_{}_M_{}/DL_converge.png'
sio.savemat(save_mat_template_training_converge.format(p['Tx_P_dB'],p['M']), {'avg_min_rate':object})
plt.savefig(save_fig_template_training_converge.format(p['Tx_P_dB'],p['M']))

test_P_tx_dB = range(p['Tx_P_dB']-20,p['Tx_P_dB']+20,2)
test_P_tx = 10**(np.asarray(test_P_tx_dB)/10)
test_avg_min_rate = np.zeros(np.size(test_P_tx))
idx_temp = 0
p['batch_size'] = 5000
NN_CoMP.batch_size = p['batch_size']
for P_idx in test_P_tx:
    NN_CoMP.P = P_idx
    H = channel_realization(p)
    BS_UE, BS_IRS, IRS_UE = construct_channel(H,p)
    BS_precoder, ISR_reflecter = NN_CoMP(BS_UE.to(device),BS_IRS.to(device),IRS_UE.to(device))
    test_avg_min_rate[idx_temp] = loss_calculator(H,p,BS_precoder.to('cpu'), ISR_reflecter.to('cpu'))
    idx_temp += 1

plt.figure(2)
plt.grid()
plt.semilogy(test_P_tx_dB,test_avg_min_rate)
plt.legend(['Proposed'])
plt.xlabel('P_tx [dB]')
plt.ylabel('Avg. min rate [bps/Hz]')
plt.show()

save_mat_template_avg_min_rate = './fig/Tx_P[dB]_{}_M_{}/DL_avg_min_rate.mat'
save_fig_template_avg_min_rate = './fig/Tx_P[dB]_{}_M_{}/DL_avg_min_rate.png'
sio.savemat(save_mat_template_avg_min_rate.format(p['Tx_P_dB'],p['M']), {'avg_min_rate':test_avg_min_rate})
plt.savefig(save_fig_template_avg_min_rate.format(p['Tx_P_dB'],p['M']))