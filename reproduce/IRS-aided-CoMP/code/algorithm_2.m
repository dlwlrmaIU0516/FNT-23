function [Phi_opt] = algorithm_2(p,H,U_opt,Q_opt,W_opt,Phi_opt)


%% Initialize r
r_idx = 0;
phi_opt = diag(Phi_opt);
error = 1;
%% Compute the maximum eigenvalue of Hardamard product A and E_tilde

% Construct A matrix
A = H.IRS_ue1'*U_opt*Q_opt*U_opt'*H.IRS_ue1;

% Construct E_tilde matrix
temp2 = zeros(p.M,p.N_t);
for idx = 1:p.num_BS
    temp = eval(sprintf('%s%g%s',strcat( 'H.bs',num2str(idx),'_IRS')))*W_opt((idx-1)*p.N_t+1:idx*p.N_t,:);
    temp2 = temp2 + temp;
end
E_tilde = temp2*temp2';

[~,Lambda] = eig(A.*transpose(E_tilde));
lambda_max = real(Lambda(1,1));
object_value = 1;
while true
    % Calculate \qv^r
    
    % Construct D matrix
    temp1 = zeros(p.N_r,p.d);
    for idx = 1:p.num_BS
        temp1 = temp1 +eval(sprintf('%s%g%s',strcat( 'H.bs',num2str(idx),'_ue1')))*W_opt((idx-1)*p.N_t+1:idx*p.N_t,:);
    end
    temp2 = zeros(p.M,p.d);
    for idx = 1:p.num_BS
        temp2 = temp2 + eval(sprintf('%s%g%s',strcat( 'H.bs',num2str(idx),'_IRS')))*W_opt((idx-1)*p.N_t+1:idx*p.N_t,:);
    end
    D = H.IRS_ue1'*U_opt*Q_opt*U_opt'*temp1*temp2';
    % Construct B matrix
    B = H.IRS_ue1'*U_opt*Q_opt*temp2';
    z=diag(D-B);
    q_r = (z-(lambda_max*eye(p.M)-A.*transpose(E_tilde))*phi_opt);
    phi_opt = -exp(j*angle(q_r));
    Phi_opt = diag(phi_opt);
    object_temp = lambda_max*p.M+2*real(phi_opt'*q_r);
    error = abs(object_value-object_temp);
    object_value = object_temp;
    
    if error<p.zeta
        break;
    end
%     disp(['Algorithm 2 error : ',num2str(error)])
end
end