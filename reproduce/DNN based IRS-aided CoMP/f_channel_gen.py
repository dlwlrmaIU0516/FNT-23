
import numpy as np

def Rician_fading(p,N_r,N_t):
    H_Los = np.zeros(shape=(p['batch_size'],N_r,N_t))
    for idx in range(p['batch_size']):
        theta_AoA = np.exp(1j*2*np.pi*(1/2)*np.reshape(np.array(range(N_r)),[N_r,1])*np.sin(np.random.uniform(1)*2*np.pi))
        theta_AoD = np.exp(1j*2*np.pi*(1/2)*np.reshape(np.array(range(N_t)),[N_t,1])*np.sin(np.random.uniform(1)*2*np.pi))
        H_Los[idx,:,:] = np.matmul(theta_AoA,theta_AoD.conj().T)

    H_NLoS = np.random.normal(size=(p['batch_size'],N_r,N_t))+1j*np.random.normal(size=(p['batch_size'],N_r,N_t))/np.sqrt(2)

    H = np.sqrt(p['Rician_f']/(p['Rician_f']+1))*H_Los + np.sqrt(1/(p['Rician_f']+1))*H_NLoS
    return H

def channel_realization(p):
    
    H = {}
    
    ue1_location = (np.random.uniform(0,1,[p['batch_size'],1,2])-0.5)*p['circle_radious']+np.sqrt(np.sum(np.asarray(p['central_point_cell_edge_users'])**2))
    ue2_location = (np.random.uniform(0,1,[p['batch_size'],1,2])-0.5)*p['circle_radious']+np.sqrt(np.sum(np.asarray(p['central_point_cell_edge_users'])**2))
    ue3_location = (np.random.uniform(0,1,[p['batch_size'],1,2])-0.5)*p['circle_radious']+np.sqrt(np.sum(np.asarray(p['central_point_cell_edge_users'])**2))

    L_loss_bs1_ue1 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_1_location']-ue1_location,axis=2)/p['d0'])**(p['alpha_bu'])
    L_loss_bs2_ue1 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_2_location']-ue1_location,axis=2)/p['d0'])**(p['alpha_bu'])
    L_loss_bs3_ue1 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_3_location']-ue1_location,axis=2)/p['d0'])**(p['alpha_bu'])

    L_loss_bs1_ue2 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_1_location']-ue2_location,axis=2)/p['d0'])**(p['alpha_bu'])
    L_loss_bs2_ue2 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_2_location']-ue2_location,axis=2)/p['d0'])**(p['alpha_bu'])
    L_loss_bs3_ue2 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_3_location']-ue2_location,axis=2)/p['d0'])**(p['alpha_bu'])

    L_loss_bs1_ue3 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_1_location']-ue3_location,axis=2)/p['d0'])**(p['alpha_bu'])
    L_loss_bs2_ue3 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_2_location']-ue3_location,axis=2)/p['d0'])**(p['alpha_bu'])
    L_loss_bs3_ue3 = 10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_3_location']-ue3_location,axis=2)/p['d0'])**(p['alpha_bu'])

    L_loss_bs1_IRS = np.tile(10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_1_location']-np.asarray(p['IRS_location']))/p['d0'])**(p['alpha_br']),(p['batch_size'],1))
    L_loss_bs2_IRS = np.tile(10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_1_location']-np.asarray(p['IRS_location']))/p['d0'])**(p['alpha_br']),(p['batch_size'],1)) 
    L_loss_bs3_IRS = np.tile(10**(p['L0_dB']/10)/( np.linalg.norm(p['BS_1_location']-np.asarray(p['IRS_location']))/p['d0'])**(p['alpha_br']),(p['batch_size'],1))

    L_loss_IRS_ue1 = 10**(p['L0_dB']/10)/( np.linalg.norm(np.asarray(p['IRS_location'])-ue1_location,axis=2)/p['d0'])**(p['alpha_ru'])
    L_loss_IRS_ue2 = 10**(p['L0_dB']/10)/( np.linalg.norm(np.asarray(p['IRS_location'])-ue2_location,axis=2)/p['d0'])**(p['alpha_ru'])
    L_loss_IRS_ue3 = 10**(p['L0_dB']/10)/( np.linalg.norm(np.asarray(p['IRS_location'])-ue3_location,axis=2)/p['d0'])**(p['alpha_ru'])

    # Rayleigh
    H['bs1_ue1'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs1_ue1),(p['batch_size'],1,1))
    H['bs2_ue1'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs2_ue1),(p['batch_size'],1,1))
    H['bs3_ue1'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs3_ue1),(p['batch_size'],1,1))

    H['bs1_ue2'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs1_ue2),(p['batch_size'],1,1))
    H['bs2_ue2'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs2_ue2),(p['batch_size'],1,1))
    H['bs3_ue2'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs3_ue2),(p['batch_size'],1,1))

    H['bs1_ue3'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs1_ue3),(p['batch_size'],1,1))
    H['bs2_ue3'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs2_ue3),(p['batch_size'],1,1))
    H['bs3_ue3'] = (np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t']))+1j*np.random.normal(size=(p['batch_size'],p['N_r'],p['N_t'])))/np.sqrt(2)*np.reshape(np.sqrt(L_loss_bs3_ue3),(p['batch_size'],1,1))

    # Rician
    H['bs1_IRS'] = Rician_fading(p,p['M'],p['N_t'])*np.reshape(np.sqrt(L_loss_bs1_IRS),(p['batch_size'],1,1))
    H['bs2_IRS'] = Rician_fading(p,p['M'],p['N_t'])*np.reshape(np.sqrt(L_loss_bs2_IRS),(p['batch_size'],1,1))
    H['bs3_IRS'] = Rician_fading(p,p['M'],p['N_t'])*np.reshape(np.sqrt(L_loss_bs3_IRS),(p['batch_size'],1,1))   
    H['IRS_ue1'] = Rician_fading(p,p['N_r'],p['M'])*np.reshape(np.sqrt(L_loss_IRS_ue1),(p['batch_size'],1,1))
    H['IRS_ue2'] = Rician_fading(p,p['N_r'],p['M'])*np.reshape(np.sqrt(L_loss_IRS_ue2),(p['batch_size'],1,1))
    H['IRS_ue3'] = Rician_fading(p,p['N_r'],p['M'])*np.reshape(np.sqrt(L_loss_IRS_ue3),(p['batch_size'],1,1))

    return H