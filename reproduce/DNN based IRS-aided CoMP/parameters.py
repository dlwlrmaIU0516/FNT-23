import numpy as np
p = {}
p['cell_type'] = 'hexagonal'
p['side_length'] = 200*np.sqrt(3)
p['num_cell'] = 1
p['simulation_scenario'] = 2 # 1-single cell-edge user , 2-Multiple cell-edge user
p['L0_dB'] = -30 # channel power gain [dB]
p['L0'] = 10**(p['L0_dB']/10)
p['d0'] = 1000 #reference distance [m]
p['num_BS'] = 3 # the number of BS
p['BS_1_location'] = [-300,0] # [x coordinate y coordinate]
p['BS_2_location'] = [300,0]
p['BS_3_location'] = [0,300*np.sqrt(3)]
p['IRS_location'] = [0,100*np.sqrt(3)]
p['central_point_cell_edge_users'] = [0,100*np.sqrt(3)]
p['circle_radious'] = 30 # [m]
p['IRS_altitude'] = 10 # [m]
p['N_t'] = 6 # the number of transmit antennas
p['N_r'] = 6 # the number of received antennas
p['M'] = 30 # the number of IRS elements [20,50,100]
p['b'] = 2 #  the number of bits to represent the resolution levels of IRS
p['K'] = 3 # the number of users

p['alpha_br'] = 2.2 # path loss exponent of BS-IRS link (Rician)
p['alpha_ru'] = 2.2 # path loss exponent of IRS-user link (Rician)
p['alpha_bu'] = 3.6 # path loss exponent of BS-user link (Rayleigh)
p['Rician_f_dB'] = 10 # Rician factor [dB]
p['Rician_f'] = 10**(p['Rician_f_dB']/10)
p['range_AoA'] = [0,2*np.pi] # arrival of angle is randomly distributed within [0 2*pi]
p['range_AoD'] = [0,2*np.pi] # arrival of departure is randomly distributed within [0 2*pi]
p['d'] = 2   # number of desired data streams
p['variance'] = -80 # noise variance [dB]
p['np'] = 10**(p['variance']*0.001/10)
p['channel_realization'] = 500 # channel realizations
p['Tx_P_dB'] = 0
p['Tx_P'] = 10**(p['Tx_P_dB']/10)
p['gamma'] = p['np']*0.1

p['batch_size'] = 512
p['lr'] = 0.01
p['iter'] = 10000