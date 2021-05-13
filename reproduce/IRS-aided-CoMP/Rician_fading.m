function [H] = Rician_fading(p,N_r,N_t)

theta_AoA = [0, rand(1,N_r-1)*2*pi];
theta_AoD = [0, rand(1,N_t-1)*2*pi];


H_Los = transpose(exp(j*2*pi*(1/2)*sin(theta_AoA)))*transpose(exp(j*2*pi*(1/2)*sin(theta_AoD))');
H_NLoS = (randn(N_r,N_t)+j*randn(N_r,N_t))/sqrt(2);

H = sqrt(p.Rician_f/(p.Rician_f+1))*H_Los + sqrt(1/(p.Rician_f+1))*H_NLoS;
end

