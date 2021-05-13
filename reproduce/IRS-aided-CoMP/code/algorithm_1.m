function [W_opt] = algorithm_1(p,H_bar,U_opt,Q_opt,H)

% Algorithm 1 Subgradient method for (P2-1).

%% Initialize \mu_n, pi_n and t
% for idx=1:p.num_BS
%     eval(sprintf('%s = %g',strcat( 'mu_',num2str(idx) ),rand(1)));
% end
% mu = rand(1,p.num_BS);
mu = [0.100962842163558   0.145619122664923];
variable_pi = rand(1,p.num_BS);
t_idx = 0;
error = 1;
object_value = 1;
gmu_value = 1;
object = [];
gmu = [];

% Calculate \hat{J}^{-1}_1
J_hat_inv_1 = inv(H_bar'*U_opt*Q_opt*U_opt'*H_bar + diag(reshape(ones(p.N_t,1).*mu,1,p.N_t*p.num_BS)));
% Calculate \hat{J}_2
J_hat_2 = [H_bar(:,1:2)'*U_opt*Q_opt;H_bar(:,3:4)'*U_opt*Q_opt];

W_opt = J_hat_inv_1*J_hat_2;
while true
    temp4 = 0;
    
    %% 1. Calculate the optimal transmit beamforming matrix using (21).
    
    
    %% Compute dual variable mu_n^{t+1} using (25).
    
    mu(1) = max(0,(mu(1)+mu(1)/100*(trace(W_opt(1:2,:)*W_opt(1:2,:)')-p.P_max)));
    mu(2) = max(0,(mu(2)+mu(2)/100*(trace(W_opt(3:4,:)*W_opt(3:4,:)')-p.P_max)));
    
    % Calculate object value of (P2-1D)
    %     for outter_idx = 1:p.num_BS
    %         for inner_idx = 1:p.num_BS
    %             temp = temp + H_bar(:,(inner_idx-1)*p.N_t+1:inner_idx*p.N_t)*W_opt((inner_idx-1)*p.N_t+1:inner_idx*p.N_t,:)...
    %                 *W_opt((outter_idx-1)*p.N_t+1:outter_idx*p.N_t,:)'*H_bar(:,(outter_idx-1)*p.N_t+1:outter_idx*p.N_t)';
    %         end
    %     end
    %
    %     for second_idx = 1:p.num_BS
    %        temp2 = temp2 + trace(Q_opt*U_opt'*H_bar(:,(second_idx-1)*p.N_t+1:second_idx*p.N_t)*W_opt((second_idx-1)*p.N_t+1:second_idx*p.N_t,:));
    %     end
    %
    %     for third_idx = 1:p.num_BS
    %         temp3 = temp3 + trace(Q_opt*W_opt((third_idx-1)*p.N_t+1:third_idx*p.N_t,:)'*H_bar(:,(third_idx-1)*p.N_t+1:third_idx*p.N_t)'*U_opt);
    %     end
    
    for forth_idx = 1:p.num_BS
        temp4 = temp4 + mu(forth_idx)*(trace(W_opt((forth_idx-1)*p.N_t+1:forth_idx*p.N_t,:)'*W_opt((forth_idx-1)*p.N_t+1:forth_idx*p.N_t,:))-p.P_max);
    end
    
    object_temp = trace(Q_opt*U_opt'*H_bar*W_opt*W_opt'*H_bar'*U_opt) - trace(Q_opt*U_opt'*H_bar*W_opt) - trace(Q_opt*W_opt'*H_bar'*U_opt);
    
    gmu_temp = object_temp+temp4;
    error = abs(gmu_temp-gmu_value);
    
    object_value = object_temp;
    gmu_value = gmu_temp;
    
    % Calculate \hat{J}^{-1}_1
    J_hat_inv_1 = inv(H_bar'*U_opt*Q_opt*U_opt'*H_bar + diag(reshape(ones(p.N_t,1).*mu,1,p.N_t*p.num_BS)));
    % Calculate \hat{J}_2
    J_hat_2 = [H_bar(:,1:2)'*U_opt*Q_opt;H_bar(:,3:4)'*U_opt*Q_opt];
    
    W_opt = J_hat_inv_1*J_hat_2;
    
    if error<p.epsilon
        break;
    end
    %     disp(['Algorithm 1 error : ',num2str(error)])
end
end