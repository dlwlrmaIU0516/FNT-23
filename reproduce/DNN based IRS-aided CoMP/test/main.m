clear,clc,close all

Nt = 2; 
Nr =4;

K = 2;

alpha = 1;
C_back = 5
P = 1;
T = 2;
% temp = (randn(Nt,T)+j*randn(Nt,T))/sqrt(2);
% S = temp./vecnorm(temp);
S =  sqrt(P/T)*eye(T);
S_til = kron(S,eye(Nr));
H_bar = ones(Nr,Nt);
for idx = 1:1000
[H] = rician(Nt,Nr,alpha,K,H_bar);
Z = (randn(Nr,T)+j*randn(Nr,T))/sqrt(2);
Y = H*S+Z-sqrt(alpha)*sqrt(K/(K+1))*H_bar*S;

H_mmse3 = Y*S'*inv(eye(Nt)/(alpha/(K+1))+S*S')+	sqrt(alpha)*sqrt(K/(K+1))*H_bar;
H_mmse4 = alpha/(K+1)*eye(Nr*Nt)*S_til'*inv(eye(Nr*Nt)+S_til*alpha/(K+1)*eye(Nr*Nt)*S_til')*Y(:)+sqrt(alpha)*sqrt(K/(K+1))*H_bar(:);


error_3 = H-H_mmse3;
error_4 = H(:)-H_mmse4;

covariance_3 = error_3(:)*error_3(:)';
covariance_4 = error_4*error_4';


temp3(idx) = covariance_3(1,1);
temp4(idx) = covariance_4(1,1);

[U,V,SS] = svd(H_bar*H_bar');
[U1,V1,SS1] = svd(H*H');

% variance_q1(idx) = det(V1+eye(Nr))^(1/Nr)/(2^(C_back/Nr)-1);
% C(idx) = log2(det(eye(Nr)+(P*H*H'+eye(Nr))/variance_q));
end


error3 = mean(temp3)
error4 = mean(temp4)

% var_temp = mean(variance_q);
variance_q = det(P/Nt*(alpha*K/(K+1)*V+alpha*Nt/(K+1)*eye(Nr))+eye(Nr))^(1/Nr)/(2^(C_back/Nr)-1);
C = log2(det(eye(Nr)+(P/Nt*H*H'+eye(Nr))/variance_q))
% error1 = var(temp1)
% error2 = var(temp2)
% error3 = var(temp3)
% MMSE_channel_est_error = mean(temp4)

Seunghwan_theorem = alpha*Nt/(T*P*alpha+Nt*(K+1))
Seunghwan_theorem1 = alpha*Nt/(P*alpha+Nt*(K+1))
% Jinkyu_theorem = alpha*Nt/(T*P+Nt*(K+1))
