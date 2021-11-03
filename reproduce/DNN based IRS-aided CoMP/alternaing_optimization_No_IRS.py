import numpy as np
import cvxpy as cp
from numpy.core.fromnumeric import var

def AO_cvx(p,H):
    W = f_initialize_W(p,H)
    Phi = f_initialize_Phi(p,H)
    Phi = np.zeros(np.shape(Phi))
    min_rate = []
    while 1:
        H_hat = f_imperfect_channel_modeling(H,p)
        H_bar = f_construct_H_bar(p,H_hat,Phi)
        U = f_update_U(p,H_bar,W)
        Q = f_update_Q(p,H_bar,W,U)
        W = f_update_W(p,H_bar,W,U,Q)
        #Phi = f_update_Phi(p,H_bar,W,U,Q,H_hat,Phi)
        min_rate = np.append(min_rate,f_min_rate(p,H,W,U,Q,Phi))
        if np.size(min_rate)>2 and np.abs(min_rate[-1]-min_rate[-2])<1e-2:
            break
    return min_rate

def f_initialize_W(p,H):
    w = []
    W = {}
    for n_idx in range(p['num_BS']):
        for n_idx in range(p['K']):
            temp_w = (np.random.normal(0,1,(p['N_t'],p['d']))+np.random.normal(0,1,(p['N_t'],p['d']))*1j)/np.sqrt(2)
            try :
                w = np.concatenate((w,temp_w/np.linalg.norm(temp_w,ord='fro')*np.sqrt((p['Tx_P']/p['K']))),axis=0)
            except np.AxisError and ValueError as e:
                w = temp_w/np.linalg.norm(temp_w,ord='fro')*np.sqrt((p['Tx_P']/p['K']))
                pass  
    W['W_1'] = w[0:p['N_t']*p['num_BS'],:]                            # W for ue1
    W['W_2'] = w[p['N_t']*p['num_BS']:p['N_t']*p['num_BS']*2,:]       # W for ue2
    W['W_3'] = w[p['N_t']*p['num_BS']*2:p['N_t']*p['num_BS']*3,:]     # W for ue3
    return W

def f_initialize_Phi(p,H):
    Phi = np.diag(np.exp(1j*np.random.rand(p['M'])*np.pi*2))
    return Phi

def f_update_U(p,H_bar,W):
    U = {}
    temp_W = W['W_1']@W['W_1'].conj().T+W['W_2']@W['W_2'].conj().T+W['W_3']@W['W_3'].conj().T
    U['U_1'] = np.linalg.inv(H_bar['H_1']@temp_W@H_bar['H_1'].conj().T+p['np']*np.eye(p['N_r']))@H_bar['H_1']@W['W_1']
    U['U_2'] = np.linalg.inv(H_bar['H_2']@temp_W@H_bar['H_2'].conj().T+p['np']*np.eye(p['N_r']))@H_bar['H_2']@W['W_2']
    U['U_3'] = np.linalg.inv(H_bar['H_3']@temp_W@H_bar['H_3'].conj().T+p['np']*np.eye(p['N_r']))@H_bar['H_3']@W['W_3']
    return U

def f_update_Q(p,H_bar,W,U):
    Q = {}
    Q['Q_1'] = np.linalg.inv(np.eye(p['d'])-W['W_1'].conj().T@H_bar['H_1'].conj().T@U['U_1'])
    Q['Q_2'] = np.linalg.inv(np.eye(p['d'])-W['W_2'].conj().T@H_bar['H_2'].conj().T@U['U_2'])
    Q['Q_3'] = np.linalg.inv(np.eye(p['d'])-W['W_3'].conj().T@H_bar['H_3'].conj().T@U['U_3'])
    return Q

def f_update_W(p,H_bar,W,U,Q):
    var_W_1_1 = cp.Variable([p['N_t'],p['d']])  # W_n,k
    var_W_1_2 = cp.Variable([p['N_t'],p['d']])
    var_W_1_3 = cp.Variable([p['N_t'],p['d']])
    var_W_2_1 = cp.Variable([p['N_t'],p['d']])
    var_W_2_2 = cp.Variable([p['N_t'],p['d']])
    var_W_2_3 = cp.Variable([p['N_t'],p['d']])
    var_W_3_1 = cp.Variable([p['N_t'],p['d']])
    var_W_3_2 = cp.Variable([p['N_t'],p['d']])
    var_W_3_3 = cp.Variable([p['N_t'],p['d']])
    var_R = cp.Variable([1])

    var_W_1 = cp.vstack([var_W_1_1,var_W_2_1,var_W_3_1]) # W_k
    var_W_2 = cp.vstack([var_W_1_2,var_W_2_2,var_W_3_2])
    var_W_3 = cp.vstack([var_W_1_3,var_W_2_3,var_W_3_3])

    var_eta_1 = cp.vstack([cp.reshape(cp.vec(var_W_1_1),(p['N_t']*p['d'],1)),
                            cp.reshape(cp.vec(var_W_1_2),(p['N_t']*p['d'],1)),
                            cp.reshape(cp.vec(var_W_1_3),(p['N_t']*p['d'],1))])

    var_eta_2 = cp.vstack([cp.reshape(cp.vec(var_W_2_1),(p['N_t']*p['d'],1)),
                            cp.reshape(cp.vec(var_W_2_2),(p['N_t']*p['d'],1)),
                            cp.reshape(cp.vec(var_W_2_3),(p['N_t']*p['d'],1))])
                            
    var_eta_3 = cp.vstack([cp.reshape(cp.vec(var_W_3_1),(p['N_t']*p['d'],1)),
                            cp.reshape(cp.vec(var_W_3_2),(p['N_t']*p['d'],1)),
                            cp.reshape(cp.vec(var_W_3_3),(p['N_t']*p['d'],1))])

    var_omega_1 = cp.vstack([cp.reshape(cp.vec(var_W_1.H@H_bar['H_1'].conj().T@U['U_1']@Q['Q_1']**(1/2)-Q['Q_1']**(1/2)),(2*p['d'],1)),
                            cp.reshape(cp.vec(var_W_2.H@H_bar['H_1'].conj().T@U['U_1']@Q['Q_1']**(1/2)),(2*p['d'],1)),
                            cp.reshape(cp.vec(var_W_3.H@H_bar['H_1'].conj().T@U['U_1']@Q['Q_1']**(1/2)),(2*p['d'],1))])

    var_omega_2 = cp.vstack([cp.reshape(cp.vec(var_W_1.H@H_bar['H_2'].conj().T@U['U_2']@Q['Q_2']**(1/2)),(2*p['d'],1)),
                            cp.reshape(cp.vec(var_W_2.H@H_bar['H_2'].conj().T@U['U_2']@Q['Q_2']**(1/2)-Q['Q_2']**(1/2)),(2*p['d'],1)),
                            cp.reshape(cp.vec(var_W_3.H@H_bar['H_2'].conj().T@U['U_2']@Q['Q_2']**(1/2)),(2*p['d'],1))])

    var_omega_3 = cp.vstack([cp.reshape(cp.vec(var_W_1.H@H_bar['H_3'].conj().T@U['U_3']@Q['Q_3']**(1/2)),(2*p['d'],1)),
                            cp.reshape(cp.vec(var_W_2.H@H_bar['H_3'].conj().T@U['U_3']@Q['Q_3']**(1/2)),(2*p['d'],1)),
                            cp.reshape(cp.vec(var_W_3.H@H_bar['H_3'].conj().T@U['U_3']@Q['Q_3']**(1/2)-Q['Q_3']**(1/2)),(2*p['d'],1))])
                            
    soc_constraint_1 = [cp.SOC(np.sqrt(p['Tx_P']),var_eta_1),
                       cp.SOC(np.sqrt(p['Tx_P']),var_eta_2),
                        cp.SOC(np.sqrt(p['Tx_P']),var_eta_3)]

    #soc_constraint_1 = [np.sqrt(p['Tx_P'])>=cp.norm(var_eta_1),
    #                    np.sqrt(p['Tx_P'])>=cp.norm(var_eta_2),
    #                    np.sqrt(p['Tx_P'])>=cp.norm(var_eta_3)]

    soc_constraint_2 = [np.linalg.slogdet(Q['Q_1'])[1]+p['d']-p['np']*np.real(np.trace(Q['Q_1']@U['U_1'].conj().T@U['U_1']))>=cp.norm(var_omega_1)**2+var_R,
                        np.linalg.slogdet(Q['Q_2'])[1]+p['d']-p['np']*np.real(np.trace(Q['Q_2']@U['U_2'].conj().T@U['U_2']))>=cp.norm(var_omega_2)**2+var_R,
                        np.linalg.slogdet(Q['Q_3'])[1]+p['d']-p['np']*np.real(np.trace(Q['Q_3']@U['U_3'].conj().T@U['U_3']))>=cp.norm(var_omega_3)**2+var_R]                    

    soc_constraints = soc_constraint_1+soc_constraint_2

    prob = cp.Problem(cp.Maximize(var_R),soc_constraints)
    #prob.solve(verbose=True)
    prob.solve()
    W['W_1'] = var_W_1.value
    W['W_2'] = var_W_2.value
    W['W_3'] = var_W_3.value
    return W


def f_update_Phi(p,H_bar,W,U,Q,H,prev_Phi):
    A_1 = H['IRS_ue1'][0,:,:].conj().T@U['U_1']@Q['Q_1']@U['U_1'].conj().T@H['IRS_ue1'][0,:,:]
    A_2 = H['IRS_ue2'][0,:,:].conj().T@U['U_2']@Q['Q_2']@U['U_2'].conj().T@H['IRS_ue2'][0,:,:]
    A_3 = H['IRS_ue3'][0,:,:].conj().T@U['U_3']@Q['Q_3']@U['U_3'].conj().T@H['IRS_ue3'][0,:,:]

    L1_1 = H['bs1_IRS'][0,:,:]@W['W_1'][0:p['N_t'],:]+H['bs2_IRS'][0,:,:]@W['W_1'][p['N_t']:p['N_t']*2,:]+H['bs3_IRS'][0,:,:]@W['W_1'][p['N_t']*2:p['N_t']*3,:]
    L1_2 = H['bs1_IRS'][0,:,:]@W['W_2'][0:p['N_t'],:]+H['bs2_IRS'][0,:,:]@W['W_2'][p['N_t']:p['N_t']*2,:]+H['bs3_IRS'][0,:,:]@W['W_2'][p['N_t']*2:p['N_t']*3,:]
    L1_3 = H['bs1_IRS'][0,:,:]@W['W_3'][0:p['N_t'],:]+H['bs2_IRS'][0,:,:]@W['W_3'][p['N_t']:p['N_t']*2,:]+H['bs3_IRS'][0,:,:]@W['W_3'][p['N_t']*2:p['N_t']*3,:]

    L2_1_1 = H['bs1_ue1'][0,:,:]@W['W_1'][0:p['N_t'],:]+H['bs2_ue1'][0,:,:]@W['W_1'][p['N_t']:p['N_t']*2,:]+H['bs3_ue1'][0,:,:]@W['W_1'][p['N_t']*2:p['N_t']*3,:]
    L2_1_2 = H['bs1_ue1'][0,:,:]@W['W_2'][0:p['N_t'],:]+H['bs2_ue1'][0,:,:]@W['W_2'][p['N_t']:p['N_t']*2,:]+H['bs3_ue1'][0,:,:]@W['W_2'][p['N_t']*2:p['N_t']*3,:]
    L2_1_3 = H['bs1_ue1'][0,:,:]@W['W_3'][0:p['N_t'],:]+H['bs2_ue1'][0,:,:]@W['W_3'][p['N_t']:p['N_t']*2,:]+H['bs3_ue1'][0,:,:]@W['W_3'][p['N_t']*2:p['N_t']*3,:]

    L2_2_1 = H['bs1_ue2'][0,:,:]@W['W_1'][0:p['N_t'],:]+H['bs2_ue2'][0,:,:]@W['W_1'][p['N_t']:p['N_t']*2,:]+H['bs3_ue2'][0,:,:]@W['W_1'][p['N_t']*2:p['N_t']*3,:]
    L2_2_2 = H['bs1_ue2'][0,:,:]@W['W_2'][0:p['N_t'],:]+H['bs2_ue2'][0,:,:]@W['W_2'][p['N_t']:p['N_t']*2,:]+H['bs3_ue2'][0,:,:]@W['W_2'][p['N_t']*2:p['N_t']*3,:]
    L2_2_3 = H['bs1_ue2'][0,:,:]@W['W_3'][0:p['N_t'],:]+H['bs2_ue2'][0,:,:]@W['W_3'][p['N_t']:p['N_t']*2,:]+H['bs3_ue2'][0,:,:]@W['W_3'][p['N_t']*2:p['N_t']*3,:]

    L2_3_1 = H['bs1_ue3'][0,:,:]@W['W_1'][0:p['N_t'],:]+H['bs2_ue3'][0,:,:]@W['W_1'][p['N_t']:p['N_t']*2,:]+H['bs3_ue3'][0,:,:]@W['W_1'][p['N_t']*2:p['N_t']*3,:]
    L2_3_2 = H['bs1_ue3'][0,:,:]@W['W_2'][0:p['N_t'],:]+H['bs2_ue3'][0,:,:]@W['W_2'][p['N_t']:p['N_t']*2,:]+H['bs3_ue3'][0,:,:]@W['W_2'][p['N_t']*2:p['N_t']*3,:]
    L2_3_3 = H['bs1_ue3'][0,:,:]@W['W_3'][0:p['N_t'],:]+H['bs2_ue3'][0,:,:]@W['W_3'][p['N_t']:p['N_t']*2,:]+H['bs3_ue3'][0,:,:]@W['W_3'][p['N_t']*2:p['N_t']*3,:]

    B_1 = H['IRS_ue1'][0,:,:].conj().T@U['U_1']@Q['Q_1']@L1_1.conj().T
    B_2 = H['IRS_ue2'][0,:,:].conj().T@U['U_2']@Q['Q_2']@L1_2.conj().T
    B_3 = H['IRS_ue3'][0,:,:].conj().T@U['U_3']@Q['Q_3']@L1_3.conj().T

    E_check = L1_1@L1_1.conj().T+L1_2@L1_2.conj().T+L1_3@L1_3.conj().T

    D_1 = H['IRS_ue1'][0,:,:].conj().T@U['U_1']@Q['Q_1']@U['U_1'].conj().T@(L2_1_1@L1_1.conj().T+L2_1_2@L1_2.conj().T+L2_1_3@L1_3.conj().T)
    D_2 = H['IRS_ue2'][0,:,:].conj().T@U['U_2']@Q['Q_2']@U['U_2'].conj().T@(L2_2_1@L1_1.conj().T+L2_2_2@L1_2.conj().T+L2_2_3@L1_3.conj().T)
    D_3 = H['IRS_ue3'][0,:,:].conj().T@U['U_3']@Q['Q_3']@U['U_3'].conj().T@(L2_3_1@L1_1.conj().T+L2_3_2@L1_2.conj().T+L2_3_3@L1_3.conj().T)

    c1_1 = np.trace(Q['Q_1']@L2_1_1.conj().T@U['U_1'])
    c1_2 = np.trace(Q['Q_2']@L2_2_2.conj().T@U['U_2'])
    c1_3 = np.trace(Q['Q_3']@L2_3_3.conj().T@U['U_3'])

    c2_1 = np.trace(L2_1_1@L2_1_1.conj().T@U['U_1']@Q['Q_1']@U['U_1'].conj().T
                    +L2_1_2@L2_1_2.conj().T@U['U_1']@Q['Q_1']@U['U_1'].conj().T
                    +L2_1_3@L2_1_3.conj().T@U['U_1']@Q['Q_1']@U['U_1'].conj().T)
    c2_2 = np.trace(L2_2_1@L2_2_1.conj().T@U['U_2']@Q['Q_2']@U['U_2'].conj().T
                    +L2_2_2@L2_2_2.conj().T@U['U_2']@Q['Q_2']@U['U_2'].conj().T
                    +L2_2_3@L2_2_3.conj().T@U['U_2']@Q['Q_2']@U['U_2'].conj().T)
    c2_3 = np.trace(L2_3_1@L2_3_1.conj().T@U['U_3']@Q['Q_3']@U['U_3'].conj().T
                    +L2_3_2@L2_3_2.conj().T@U['U_3']@Q['Q_3']@U['U_3'].conj().T
                    +L2_3_3@L2_3_3.conj().T@U['U_3']@Q['Q_3']@U['U_3'].conj().T)

    const_1 = np.real(np.linalg.slogdet(Q['Q_1'])[1]+p['d']+2*np.real(c1_1)-c2_1-np.trace(Q['Q_1']@(p['np']*U['U_1'].conj().T@U['U_1']+np.eye(p['d']))))
    const_2 = np.real(np.linalg.slogdet(Q['Q_2'])[1]+p['d']+2*np.real(c1_2)-c2_2-np.trace(Q['Q_2']@(p['np']*U['U_2'].conj().T@U['U_2']+np.eye(p['d']))))
    const_3 = np.real(np.linalg.slogdet(Q['Q_3'])[1]+p['d']+2*np.real(c1_3)-c2_3-np.trace(Q['Q_3']@(p['np']*U['U_3'].conj().T@U['U_3']+np.eye(p['d']))))

    z_1 = np.reshape(np.diag(D_1-B_1),(p['M'],1))
    z_2 = np.reshape(np.diag(D_2-B_2),(p['M'],1))
    z_3 = np.reshape(np.diag(D_3-B_3),(p['M'],1))

    Psi_1 = np.real(np.concatenate((np.concatenate((np.multiply(A_1,E_check.T),z_1),1),np.concatenate((z_1.conj().T,np.zeros((1,1))),1)),0))
    Psi_2 = np.real(np.concatenate((np.concatenate((np.multiply(A_2,E_check.T),z_2),1),np.concatenate((z_2.conj().T,np.zeros((1,1))),1)),0))
    Psi_3 = np.real(np.concatenate((np.concatenate((np.multiply(A_3,E_check.T),z_3),1),np.concatenate((z_3.conj().T,np.zeros((1,1))),1)),0))

    #var_theta = cp.Variable((p['M']+1,1),complex=True)
    #var_Theta = var_theta@var_theta.H
    var_Theta = cp.Variable((p['M']+1,p['M']+1),complex=True)
    var_R = cp.Variable([1])

    constraint_1 = [cp.real(cp.trace(Psi_1@var_Theta))<=const_1-var_R,
                    cp.real(cp.trace(Psi_2@var_Theta))<=const_2-var_R,
                    cp.real(cp.trace(Psi_3@var_Theta))<=const_3-var_R]

    constraint_2 = [var_Theta >> 0]

    constraint_3 = [cp.diag(var_Theta)==np.ones((p['M']+1,))]

    constraints = constraint_1 + constraint_2 + constraint_3


    prob = cp.Problem(cp.Maximize(var_R),constraints)
    prob.solve()
    
    Phi = f_randomization(p,var_Theta.value)

    final_Phi = f_heuristic_update(p,W,U,Q,H,prev_Phi,Phi)
    return final_Phi


def f_construct_H_bar(p,H,Phi):
    H_bar = {}
    H_1_1 = H['bs1_ue1'][0,:,:]+H['IRS_ue1'][0,:,:]@Phi@H['bs1_IRS'][0,:,:]
    H_1_2 = H['bs1_ue2'][0,:,:]+H['IRS_ue2'][0,:,:]@Phi@H['bs1_IRS'][0,:,:]
    H_1_3 = H['bs1_ue3'][0,:,:]+H['IRS_ue3'][0,:,:]@Phi@H['bs1_IRS'][0,:,:]

    H_2_1 = H['bs2_ue1'][0,:,:]+H['IRS_ue1'][0,:,:]@Phi@H['bs2_IRS'][0,:,:]
    H_2_2 = H['bs2_ue2'][0,:,:]+H['IRS_ue2'][0,:,:]@Phi@H['bs2_IRS'][0,:,:]
    H_2_3 = H['bs2_ue3'][0,:,:]+H['IRS_ue3'][0,:,:]@Phi@H['bs2_IRS'][0,:,:]

    H_3_1 = H['bs3_ue1'][0,:,:]+H['IRS_ue1'][0,:,:]@Phi@H['bs3_IRS'][0,:,:]
    H_3_2 = H['bs3_ue2'][0,:,:]+H['IRS_ue2'][0,:,:]@Phi@H['bs3_IRS'][0,:,:]
    H_3_3 = H['bs3_ue3'][0,:,:]+H['IRS_ue3'][0,:,:]@Phi@H['bs3_IRS'][0,:,:]

    H_bar['H_1'] = np.concatenate((H_1_1 , H_2_1 , H_3_1),1)
    H_bar['H_2'] = np.concatenate((H_1_2 , H_2_2 , H_3_2),1)
    H_bar['H_3'] = np.concatenate((H_1_3 , H_2_3 , H_3_3),1)

    return H_bar

def f_min_rate(p,H_bar,W,U,Q,Phi):
    rate = f_rate(p,H_bar,W,U,Q,Phi)
    min_rate = np.min([rate['rate_1'],rate['rate_2'],rate['rate_3']])
    return min_rate

def f_rate(p,H,W,U,Q,Phi):
    rate = {}
    H_bar = f_construct_H_bar(p,H,Phi)
    temp_W_1 = W['W_2']@W['W_2'].conj().T+W['W_3']@W['W_3'].conj().T
    temp_W_2 = W['W_1']@W['W_1'].conj().T+W['W_3']@W['W_3'].conj().T
    temp_W_3 = W['W_1']@W['W_1'].conj().T+W['W_2']@W['W_2'].conj().T
    temp_F_1 = H_bar['H_1']@temp_W_1@H_bar['H_1'].conj().T+p['np']*np.eye(p['N_r'])
    temp_F_2 = H_bar['H_2']@temp_W_2@H_bar['H_2'].conj().T+p['np']*np.eye(p['N_r'])
    temp_F_3 = H_bar['H_3']@temp_W_3@H_bar['H_3'].conj().T+p['np']*np.eye(p['N_r'])
    rate['rate_1'] = np.linalg.slogdet(np.eye(p['N_r'])+H_bar['H_1']@W['W_1']@W['W_1'].conj().T@H_bar['H_1'].conj().T@np.linalg.inv(temp_F_1))[1]
    rate['rate_2'] = np.linalg.slogdet(np.eye(p['N_r'])+H_bar['H_2']@W['W_2']@W['W_2'].conj().T@H_bar['H_2'].conj().T@np.linalg.inv(temp_F_2))[1]
    rate['rate_3'] = np.linalg.slogdet(np.eye(p['N_r'])+H_bar['H_3']@W['W_3']@W['W_3'].conj().T@H_bar['H_3'].conj().T@np.linalg.inv(temp_F_3))[1]
    return rate

def f_randomization(p,Pi):
    # ref : Transmit Beamforming for Physical-Layer Multicasting 
    svd_U,svd_sigma,svd_V_H = np.linalg.svd(Pi)
    v = np.random.randn(p['M']+1,1)
    phi_thilde = svd_U@np.diag(svd_sigma)**(1/2)@v
    phi_bar = np.exp(1j*np.angle(phi_thilde/phi_thilde[p['M'],0]))
    phi = phi_bar[0:p['M'],0]
    Phi = np.diag(phi)
    return Phi

def f_heuristic_update(p,W,U,Q,H,prev,current):
    prev_rate = f_min_rate(p,H,W,U,Q,prev)
    current_rate = f_min_rate(p,H,W,U,Q,current)

    if current_rate>=prev_rate:
        final_Phi=current
    else:
        final_Phi=prev
    return final_Phi

def f_imperfect_channel_modeling(H,p):
    H_hat = {}

    H_hat['bs1_ue1'] = H['bs1_ue1'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs1_ue1']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs1_ue1']))*1j)/np.sqrt(2)
    H_hat['bs2_ue1'] = H['bs2_ue1'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs2_ue1']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs2_ue1']))*1j)/np.sqrt(2)
    H_hat['bs3_ue1'] = H['bs3_ue1'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs3_ue1']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs3_ue1']))*1j)/np.sqrt(2)

    H_hat['bs1_ue2'] = H['bs1_ue2'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs1_ue2']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs1_ue2']))*1j)/np.sqrt(2)
    H_hat['bs2_ue2'] = H['bs2_ue2'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs2_ue2']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs2_ue2']))*1j)/np.sqrt(2)
    H_hat['bs3_ue2'] = H['bs3_ue2'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs3_ue2']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs3_ue2']))*1j)/np.sqrt(2)

    H_hat['bs1_ue3'] = H['bs1_ue3'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs1_ue3']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs1_ue3']))*1j)/np.sqrt(2)
    H_hat['bs2_ue3'] = H['bs2_ue3'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs2_ue3']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs2_ue3']))*1j)/np.sqrt(2)
    H_hat['bs3_ue3'] = H['bs3_ue3'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs3_ue3']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs3_ue3']))*1j)/np.sqrt(2)

    # Rician
    H_hat['bs1_IRS'] = H['bs1_IRS'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs1_IRS']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs1_IRS']))*1j)/np.sqrt(2)
    H_hat['bs2_IRS'] = H['bs2_IRS'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs2_IRS']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs2_IRS']))*1j)/np.sqrt(2)
    H_hat['bs3_IRS'] = H['bs3_IRS'] + (np.random.normal(0,p['gamma'] ,np.shape(H['bs3_IRS']))+np.random.normal(0,p['gamma'] ,np.shape(H['bs3_IRS']))*1j)/np.sqrt(2)
    H_hat['IRS_ue1'] = H['IRS_ue1'] + (np.random.normal(0,p['gamma'] ,np.shape(H['IRS_ue1']))+np.random.normal(0,p['gamma'] ,np.shape(H['IRS_ue1']))*1j)/np.sqrt(2)
    H_hat['IRS_ue2'] = H['IRS_ue2'] + (np.random.normal(0,p['gamma'] ,np.shape(H['IRS_ue2']))+np.random.normal(0,p['gamma'] ,np.shape(H['IRS_ue2']))*1j)/np.sqrt(2)
    H_hat['IRS_ue3'] = H['IRS_ue3'] + (np.random.normal(0,p['gamma'] ,np.shape(H['IRS_ue3']))+np.random.normal(0,p['gamma'] ,np.shape(H['IRS_ue3']))*1j)/np.sqrt(2)


    return H_hat