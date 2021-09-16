import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class IRS_CoMP_Net(nn.Module):

    def __init__(self,N_t,N_r,M,N_BS,K,d,batch_size,P):
        super(IRS_CoMP_Net, self).__init__()
        self.batch_size = batch_size
        self.N_t = N_t
        self.N_r = N_r
        self.M = M
        self.N_BS = N_BS
        self.K = K
        self.d = d
        self.P = P

        self.fc1_1 = nn.Linear(self.N_t*self.N_r*2*self.N_BS*self.K,1298) # direct link
        self.fc1_2 = nn.Linear(self.N_t*self.M*2*self.N_BS,7200) # reflect link for BS-ISR
        self.fc1_3 = nn.Linear(self.M*self.N_r*2*self.K,7200) # reflect link for ISR-User

        self.fc2 = nn.Linear(7200*2+1298,512)
        self.fc3 = nn.Linear(512,512)

        self.fc4_1 = nn.Linear(512,256)
        self.fc4_2 = nn.Linear(512,256)

        self.fc5_1 = nn.Linear(256,N_BS*d*K*N_t*2)
        self.fc5_2 = nn.Linear(256,M*2)
        
    def IRS_normalization(self, y2):
        temp = torch.reshape(y2,(self.batch_size,self.M,2))
        temp2 = torch.sqrt(temp[:,:,0]**2+temp[:,:,1]**2)
        return torch.diag_embed(temp[:,:,0]/temp2+1j*temp[:,:,1]/temp2, offset=0, dim1=-2, dim2=-1)

    def BS_normalization(self, y1):
        temp0 = torch.reshape(y1,(self.batch_size,self.N_t*2,self.N_BS*self.K*self.d))
        temp_1 = temp0[:,:,0:self.d*self.K]
        temp_2 = temp0[:,:,self.d*self.K:self.d*self.K*2]
        temp_3 = temp0[:,:,self.d*self.K*2:self.d*self.K*3]

        temp1 = torch.sqrt(torch.tensor(self.P))*temp_1/torch.reshape(torch.norm(temp_1,p='fro',dim=(1,2)),(self.batch_size,-1,1))
        w1 = temp1[:,0:self.N_t,:]+1j*temp1[:,self.N_t:self.N_t*2,:]

        temp2 = torch.sqrt(torch.tensor(self.P))*temp_2/torch.reshape(torch.norm(temp_2,p='fro',dim=(1,2)),(self.batch_size,-1,1))
        w2 = temp2[:,0:self.N_t,:]+1j*temp2[:,self.N_t:self.N_t*2,:]

        temp3 = torch.sqrt(torch.tensor(self.P))*temp_3/torch.reshape(torch.norm(temp_3,p='fro',dim=(1,2)),(self.batch_size,-1,1))
        w3 = temp3[:,0:self.N_t,:]+1j*temp3[:,self.N_t:self.N_t*2,:]

        return torch.cat([w1,w2,w3],2)

    def forward(self, x1,x2,x3):

        x1 = F.relu(self.fc1_1(x1))
        x2 = F.relu(self.fc1_2(x2))
        x3 = F.relu(self.fc1_3(x3))

        x = F.relu(self.fc2(torch.cat([x1,x2,x3],1)))
        x = F.tanh(self.fc3(x))

        y1 = self.fc4_1(x)
        y2 = self.fc4_2(x)

        y1 = self.fc5_1(y1)
        y2 = self.fc5_2(y2)

        BS = self.BS_normalization(y1)
        IRS = self.IRS_normalization(y2)

        return BS,IRS


def construct_channel(H,p):

    H_bs1_ue1 = np.concatenate((np.real(H['bs1_ue1']),np.imag(H['bs1_ue1'])),axis=1)
    H_bs1_ue2 = np.concatenate((np.real(H['bs1_ue2']),np.imag(H['bs1_ue2'])),axis=1)
    H_bs1_ue3 = np.concatenate((np.real(H['bs1_ue3']),np.imag(H['bs1_ue3'])),axis=1)
    H_bs2_ue1 = np.concatenate((np.real(H['bs2_ue1']),np.imag(H['bs2_ue1'])),axis=1)
    H_bs2_ue2 = np.concatenate((np.real(H['bs2_ue2']),np.imag(H['bs2_ue2'])),axis=1)
    H_bs2_ue3 = np.concatenate((np.real(H['bs2_ue3']),np.imag(H['bs2_ue3'])),axis=1)
    H_bs3_ue1 = np.concatenate((np.real(H['bs3_ue1']),np.imag(H['bs3_ue1'])),axis=1)
    H_bs3_ue2 = np.concatenate((np.real(H['bs3_ue2']),np.imag(H['bs3_ue2'])),axis=1)
    H_bs3_ue3 = np.concatenate((np.real(H['bs3_ue3']),np.imag(H['bs3_ue3'])),axis=1)
    
    np_bs_ue =  np.concatenate((H_bs1_ue1,
                                H_bs1_ue2,
                                H_bs1_ue3,
                                H_bs2_ue1,
                                H_bs2_ue2,
                                H_bs2_ue3,
                                H_bs3_ue1,
                                H_bs3_ue2,
                                H_bs3_ue3),axis=1)

    BS_UE = torch.reshape(torch.from_numpy(np_bs_ue).type(torch.FloatTensor),(p['batch_size'],-1))

    H_bs1_IRS = np.concatenate((np.real(H['bs1_IRS']),np.imag(H['bs1_IRS'])),axis=1)
    H_bs2_IRS = np.concatenate((np.real(H['bs2_IRS']),np.imag(H['bs2_IRS'])),axis=1)
    H_bs3_IRS = np.concatenate((np.real(H['bs3_IRS']),np.imag(H['bs3_IRS'])),axis=1)

    np_bs_IRS =  np.concatenate((H_bs1_IRS,
                                H_bs2_IRS,
                                H_bs3_IRS),axis=1)

    BS_IRS = torch.reshape(torch.from_numpy(np_bs_IRS).type(torch.FloatTensor),(p['batch_size'],-1))

    H_IRS_UE1 = np.concatenate((np.real(H['IRS_ue1']),np.imag(H['IRS_ue1'])),axis=1)
    H_IRS_UE2 = np.concatenate((np.real(H['IRS_ue2']),np.imag(H['IRS_ue2'])),axis=1)
    H_IRS_UE3 = np.concatenate((np.real(H['IRS_ue3']),np.imag(H['IRS_ue3'])),axis=1)

    np_IRS_UE =  np.concatenate((H_IRS_UE1,
                                H_IRS_UE2,
                                H_IRS_UE3),axis=1)

    IRS_UE = torch.reshape(torch.from_numpy(np_IRS_UE).type(torch.FloatTensor),(p['batch_size'],-1))
                                
    return BS_UE, BS_IRS, IRS_UE

def loss_calculator(H,p,BS,IRS):

    H_bar_1_1 = torch.from_numpy(H['bs1_ue1']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue1']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs1_IRS']).type(torch.cfloat)))
    H_bar_1_2 = torch.from_numpy(H['bs1_ue2']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue2']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs1_IRS']).type(torch.cfloat)))
    H_bar_1_3 = torch.from_numpy(H['bs1_ue3']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue3']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs1_IRS']).type(torch.cfloat)))

    H_bar_2_1 = torch.from_numpy(H['bs2_ue1']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue1']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs2_IRS']).type(torch.cfloat)))
    H_bar_2_2 = torch.from_numpy(H['bs2_ue2']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue2']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs2_IRS']).type(torch.cfloat)))
    H_bar_2_3 = torch.from_numpy(H['bs2_ue3']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue3']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs2_IRS']).type(torch.cfloat)))
    
    H_bar_3_1 = torch.from_numpy(H['bs3_ue1']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue1']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs3_IRS']).type(torch.cfloat)))
    H_bar_3_2 = torch.from_numpy(H['bs3_ue2']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue2']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs3_IRS']).type(torch.cfloat)))
    H_bar_3_3 = torch.from_numpy(H['bs3_ue3']).type(torch.cfloat)+torch.matmul(torch.from_numpy(H['IRS_ue3']).type(torch.cfloat),torch.matmul(IRS,torch.from_numpy(H['bs3_IRS']).type(torch.cfloat)))

    H_bar_1 = torch.cat([H_bar_1_1,H_bar_2_1,H_bar_3_1],2)
    H_bar_2 = torch.cat([H_bar_1_2,H_bar_2_2,H_bar_3_2],2)
    H_bar_3 = torch.cat([H_bar_1_3,H_bar_2_3,H_bar_3_3],2)

    W_1 = torch.cat((BS[:,:,0:2],BS[:,:,2:4],BS[:,:,4:6]),1)
    W_2 = torch.cat((BS[:,:,6:8],BS[:,:,8:10],BS[:,:,10:12]),1)
    W_3 = torch.cat((BS[:,:,12:14],BS[:,:,14:16],BS[:,:,16:18]),1)

    channel_covar_1 = torch.matmul(torch.matmul(H_bar_1,W_1),torch.transpose(torch.conj(torch.matmul(H_bar_1,W_1)),1,2))
    inter_covar_1 = torch.matmul(torch.matmul(H_bar_1,(torch.matmul(W_2,torch.transpose(torch.conj(W_2),1,2))+torch.matmul(W_3,torch.transpose(torch.conj(W_3),1,2)))),torch.transpose(torch.conj(H_bar_1),1,2))+torch.eye(torch.tensor(p['N_r']))*torch.tensor(p['np'])

    channel_covar_2 = torch.matmul(torch.matmul(H_bar_2,W_2),torch.transpose(torch.conj(torch.matmul(H_bar_2,W_2)),1,2))
    inter_covar_2 = torch.matmul(torch.matmul(H_bar_2,(torch.matmul(W_1,torch.transpose(torch.conj(W_1),1,2))+torch.matmul(W_3,torch.transpose(torch.conj(W_3),1,2)))),torch.transpose(torch.conj(H_bar_2),1,2))+torch.eye(torch.tensor(p['N_r']))*torch.tensor(p['np'])

    channel_covar_3 = torch.matmul(torch.matmul(H_bar_3,W_3),torch.transpose(torch.conj(torch.matmul(H_bar_3,W_3)),1,2))
    inter_covar_3 = torch.matmul(torch.matmul(H_bar_3,(torch.matmul(W_1,torch.transpose(torch.conj(W_1),1,2))+torch.matmul(W_2,torch.transpose(torch.conj(W_2),1,2)))),torch.transpose(torch.conj(H_bar_3),1,2))+torch.eye(torch.tensor(p['N_r']))*torch.tensor(p['np'])


    UE1_rate = torch.reshape(torch.linalg.slogdet(torch.eye(torch.tensor(p['N_r']))+torch.matmul(channel_covar_1,torch.inverse(inter_covar_1)))[1],(p['batch_size'],-1))
    UE2_rate = torch.reshape(torch.linalg.slogdet(torch.eye(torch.tensor(p['N_r']))+torch.matmul(channel_covar_2,torch.inverse(inter_covar_2)))[1],(p['batch_size'],-1))
    UE3_rate = torch.reshape(torch.linalg.slogdet(torch.eye(torch.tensor(p['N_r']))+torch.matmul(channel_covar_3,torch.inverse(inter_covar_3)))[1],(p['batch_size'],-1))

    
    return torch.mean(torch.min(torch.cat([UE1_rate,UE2_rate,UE3_rate],1),1)[0])

def train(loss,optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
