clear,clc,close all

Nt = 10;
Nr = 10;

K = 1;

alpha = 100;
C_back = 2^2;
P = 10;
% T = 30;

for Nt = 1:30
    for Nr = 1:30
        T= Nt;
    H_bar = (randn(Nr,Nt)+j*randn(Nr,Nt))/sqrt(2);
        
        %% CFE
        for idx = 1 : 1
            [H] = rician(Nt,Nr,alpha,K,H_bar);
            [U,V,S] = svd(H_bar*H_bar');
            
            variance_q  = det(P/Nt/T*(alpha*K/(K+1)*V+alpha*Nt/(K+1)*eye(Nr))+eye(Nr))^(1/Nr)/(2^(C_back/Nr)-1);
            
            var_CFE(idx) = alpha*Nt*(1+variance_q)/(alpha*P+Nt*(1+variance_q)*(1+K));

        end
        CFE(Nt,Nr) = mean(var_CFE);
        
        
        %% ECF
        for idx = 1 : 1
            var_h_tilde = alpha^2*P/(K+1)/(alpha*P+Nt*(1+K));
            var_ECF(idx) = var_h_tilde*2^(-T*C_back/(Nr*Nt));
        end
        ECF(Nt,Nr) = mean(var_ECF);
        
    end
    
end
figure(1)
mesh(1:30,1:30,CFE,'facecolor','r');
% ylabel('Nt')
% xlabel('Nr')
% zlabel('CFE_{error}')
hold on
% figure(2)
mesh(1:30,1:30,ECF,'facecolor','b');
ylabel('Nt')
xlabel('Nr')
% ylabel('alpha')
% xlabel('C')
% zlabel('ECF_{error}')
zlabel('Error')
% legend('CFE','ECF')
title(['P=',num2str(P),',K=',num2str(K),', alpha=',num2str(alpha),', C =',num2str(C_back),', T=',num2str(T)])