import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import os
import warnings
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings(action='ignore')
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    device = torch.device("cuda")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")
print('cuda index:', torch.cuda.current_device())
print('number of gpu:', torch.cuda.device_count())
print('graphic name:', torch.cuda.get_device_name())
#cuda = torch.device('cuda')
print(device)

p = {}
p['batch_size'] = 1000
p['N_r_range'] = 31
p['N_t_range'] = 31
p['C'] = 4
p['SNR_dB'] = 10
p['SNR'] = 10**(p['SNR_dB']/10)
p['lr'] = 0.01
p['alpha'] = torch.tensor(1.)
p['K'] = torch.tensor(2.)
p['T'] = 30

class Coordination_net(nn.Module):

    def __init__(self,p):
        super(Coordination_net, self).__init__()
        self.batch_size = p['batch_size']
        self.N_t = p['N_t']
        self.N_r = p['N_r']
        self.C = p['C']

        self.fc1 = nn.Linear(self.N_t*self.N_r*2,500) # direct link
        self.fc2 = nn.Linear(500,500)
        self.fc3 = nn.Linear(500,500)
        self.fc4 = nn.Linear(500,500)
        self.fc5 = nn.Linear(500,self.C)

        self.fc6 = nn.Linear(self.C,500)
        self.fc7 = nn.Linear(500,500)
        self.fc8 = nn.Linear(500,500)
        self.fc9 = nn.Linear(500,500)
        self.fc10 = nn.Linear(500,self.N_t*self.N_r*2)

    def forward(self, x):
        
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.tanh(self.fc3(x2))
        x4 = F.tanh(self.fc4(x3))
        x5 = F.tanh(self.fc5(x4))
        
        with torch.no_grad():
            b_x3 = self.b_layer(x5)-x5
        
        x6 = self.fc6(b_x3+x5)
        x7 = self.fc7(x6)
        x8 = self.fc8(x7)
        x9 = self.fc9(x8)
        y = self.fc10(x9)

        return self.b_layer(x5),y

    def b_layer(self,x):
        return 2*torch.bernoulli((1+x)/2)-1

def train(optimizer,loss_value):
    optimizer.zero_grad() 
    loss_value.backward() 
    optimizer.step()

loss_temp = np.zeros([p['N_t_range'],p['N_r_range']])
for Nt_idx in range(0,p['N_t_range'],10):
    for Nr_idx in range(0,p['N_r_range'],10):
        p['N_r'] = Nr_idx+1
        p['N_t'] = Nt_idx+1
        p['T'] = p['N_t']
        DNN = Coordination_net(p).to(device)
        optimizer = optim.SGD(DNN.parameters(), lr=p['lr'])
        loss = nn.MSELoss().to(device)
        H_bar = torch.ones(size=(p['batch_size'],p['N_r']*p['N_t']*2))
        for idx in range(30000):
            H = torch.sqrt(p['alpha'])*torch.sqrt(p['K']/(p['K']+1))*H_bar\
            +torch.sqrt(p['alpha'])*torch.sqrt(1/(p['K']+1))*torch.normal(mean = 0, std = 1, size=(p['batch_size'],p['N_r']*p['N_t']*2))/torch.sqrt(torch.tensor(2))

            N = torch.normal(mean = 0, std = 1, size=(p['batch_size'],p['N_r']*p['N_t']*2))/torch.sqrt(torch.tensor(2))

            received_pilot = torch.sqrt(torch.tensor(p['SNR'])/p['N_t'])*H+N

            b_x3, estimate = DNN(received_pilot.to(device))
            loss_value = loss(H.to(device),estimate)
            train(optimizer,loss_value)
            call_loss = loss_value.to('cpu')
            error_temp = (H.to(device)-estimate)**2
            error = torch.mean(error_temp[:,0:p['N_r']*p['N_t']]+error_temp[:,p['N_r']*p['N_t']:p['N_r']*p['N_t']*2])
            if idx % 100 == 0:
                print('Iter ',idx,',N_t ',p['N_t'],',N_r ',p['N_r'],': ',call_loss.detach().numpy(),'error :',error.to('cpu').detach().numpy())

        # Test
        p['batch_size'] = 10000
        DNN.batch_size = p['batch_size']
        H_bar = torch.ones(size=(p['batch_size'],p['N_r']*p['N_t']*2))
        H = torch.sqrt(p['alpha'])*torch.sqrt(p['K']/(p['K']+1))*H_bar\
        +torch.sqrt(p['alpha'])*torch.sqrt(1/(p['K']+1))*torch.normal(mean = 0, std = 1, size=(p['batch_size'],p['N_r']*p['N_t']*2))/torch.sqrt(torch.tensor(2))

        N = torch.normal(mean = 0, std = 1, size=(p['batch_size'],p['N_r']*p['N_t']*2))/torch.sqrt(torch.tensor(2))

        received_pilot = torch.sqrt(torch.tensor(p['SNR'])/p['N_t'])*H+N

        b_x3, estimate = DNN(received_pilot.to(device))
        loss_value = loss(H.to(device),estimate)
        test_error_temp = (H.to(device)-estimate)**2
        test_error = torch.mean(test_error_temp[:,0:p['N_r']*p['N_t']]+test_error_temp[:,p['N_r']*p['N_t']:p['N_r']*p['N_t']*2])
        p['batch_size'] = 1000

        print(b_x3)
        print('Test MSE loss : ',loss_value.to('cpu'))
        loss_temp[Nt_idx,Nr_idx] = test_error

save_mat_template_MSE = './fig/P[dB]_{}_K_{}_alpha_{}_C_{}_T_{}/MSE.mat'
path = './fig/P[dB]_{}_K_{}_alpha_{}_C_{}_T_{}'
try:
    if not(os.path.isdir(path.format(p['SNR_dB'],p['K'],p['alpha'],p['C'],p['T']))):
        os.makedirs(os.path.join(path.format(p['SNR_dB'],p['K'],p['alpha'],p['C'],p['T'])))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise
sio.savemat(save_mat_template_MSE.format(p['SNR_dB'],p['K'],p['alpha'],p['C'],p['T']), {'MSE':loss_temp})
