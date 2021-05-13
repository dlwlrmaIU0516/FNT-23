function [objective_value] = algorithm_3(p,H)


%% 1. Initialization \Wv_n and \mathbf{\Phi}

while true
    
    %% 2. Calculte \Wv^{opt} from (15)
    U_opt = ??;
    
    %% 3. Calculte \Qv^{opt} from (16)
    Q_opt = ??;
    
    %% 4. Calculate \Wv^{opt}_n from Algorithm 1.
    [W_opt] = algorithm_1(??);
    
    %% 5. Calculate \Phi^{opt}_n from Algorithm 2.
    [Phi_opt] = algorithm_2(??);
    
    %% Stopping criteria
    if ??
        break;
    end
    %     disp(['Algorithm 3 error : ',num2str(error)]);
end
end

