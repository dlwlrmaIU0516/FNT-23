clear,clc,close all

Nt_range = 1:30;
Nr_range = 1:30;

K = 2;

alpha = 1;
C_back = 4;
P = 10^(10/10);
% T = 30;

for Nt = Nt_range
    for Nr = Nr_range
    H_bar = (ones(Nr,Nt)+j*ones(Nr,Nt));
    T = Nt;
        
        %% CFE
        for idx = 1 : 1000
            [H] = rician(Nt,Nr,alpha,K,H_bar);
            [U,V,S] = svd(H_bar*H_bar');
            
            variance_q  = det(P/Nt*(alpha*K/(K+1)*V+alpha*Nt/(K+1)*eye(Nr))+eye(Nr))^(1/Nr)/(2^(C_back/Nr)-1);
            
            var_CFE(idx) = alpha*Nt*(1+variance_q)/(alpha*P*T+Nt*(1+variance_q)*(1+K));

        end
        CFE(Nt,Nr) = mean(var_CFE);
        
        
        %% ECF
        for idx = 1 : 1
            var_h_tilde = alpha^2*P*T/(K+1)/(alpha*P*T+Nt*(1+K));
            var_ECF(idx) = var_h_tilde*2^(-T*C_back/(Nr*Nt));
        end
        ECF(Nt,Nr) = mean(var_ECF);
        
    end
    
end
figure(1)
mesh(Nr_range,Nt_range,CFE,'facecolor','r');
% ylabel('Nt')
% xlabel('Nr')
% zlabel('CFE_{error}')
hold on
% figure(2)
mesh(Nr_range,Nt_range,ECF,'facecolor','b');
ylabel('Nt')
xlabel('Nr')
% ylabel('alpha')
% xlabel('C')
% zlabel('ECF_{error}')
zlabel('Error')
% legend('CFE','ECF')
title(['P=',num2str(P),',K=',num2str(K),', alpha=',num2str(alpha),', C =',num2str(C_back),', T=Nt'])
legend(['CFE'],['ECF'],['DNN'])