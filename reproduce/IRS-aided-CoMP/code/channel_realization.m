function [H] = channel_realization(p)

% Rician fading reference : https://arxiv.org/pdf/1907.10864.pdf

% H.bs1_ue1 : H1
% H.bs2_ue1 : H2
% H.bs1_IRS : G1
% H.bs2_IRS : G1
% H.IRS_ue1 : Hr
if p.simulation_scenario == 'single cell-edge user'
    
    L_loss_bs1_ue1 = 10^(p.L0_dB/10)/(norm(p.BS_1_location-p.cell_edge_user_1_location)/p.d0)^(p.alpha_bu);
    L_loss_bs2_ue1 = 10^(p.L0_dB/10)/(norm(p.BS_2_location-p.cell_edge_user_1_location)/p.d0)^(p.alpha_bu);
    L_loss_bs1_IRS = 10^(p.L0_dB/10)/(norm(p.BS_1_location-p.IRS_location)/p.d0)^(p.alpha_br);
    L_loss_bs2_IRS = 10^(p.L0_dB/10)/(norm(p.BS_2_location-p.IRS_location)/p.d0)^(p.alpha_br); 
    L_loss_IRS_ue1 = 10^(p.L0_dB/10)/(norm(p.IRS_location-p.cell_edge_user_1_location)/p.d0)^(p.alpha_ru);
    
    % Rayleigh
    H.bs1_ue1 = (randn(p.N_r,p.N_t)+j*randn(p.N_r,p.N_t))/sqrt(2)*sqrt(L_loss_bs1_ue1);
    % Rayleigh
    H.bs2_ue1 = (randn(p.N_r,p.N_t)+j*randn(p.N_r,p.N_t))/sqrt(2)*sqrt(L_loss_bs2_ue1);
    % Rician
    H.bs1_IRS =  Rician_fading(p,p.M,p.N_t)*sqrt(L_loss_bs1_IRS);
    % Rician
    H.bs2_IRS = Rician_fading(p,p.M,p.N_t)*sqrt(L_loss_bs2_IRS);
    % Rician
    H.IRS_ue1 = Rician_fading(p,p.N_r,p.M)*sqrt(L_loss_IRS_ue1);

end

