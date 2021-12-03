function [H] = rician(N_t,N_r,alpha,K,H_bar)
% H_bar = (randn(N_r,N_t)+j*randn(N_r,N_t))/sqrt(2);
% H_bar = zeros(N_r,N_t);
H1 = (randn(N_r,N_t)+j*randn(N_r,N_t))/sqrt(2);
H = sqrt(alpha)*(sqrt(K/(K+1))*H_bar+sqrt(1/(K+1))*H1);
% H = sqrt(1/(K+1))*H1;
end
