clc;
clear;
cvx_solver MOSEK
Rt2 = 5;
Rtc = 5;
eta2 = 2^Rt2-1;
etac = 2^Rtc-1;

ANTENNA = 3;
sigma_dbm = -94;
SIGMA =10^(sigma_dbm/10-3);
P0_dbm = -30;
P0 = 10^(P0_dbm/10-3);

d1 = 10;
d2 = 10;
d3 = 10;
d4 = 10;

kappa = 1;
path = 1.5;
path2 = 1.8;
path3 = 1.8;


h1 = sqrt(kappa/(1+kappa)) + sqrt(1/(1+kappa))* (((randn(1,ANTENNA) + 1i*randn)/sqrt(2))/(sqrt(d1^path2)));
h2 = sqrt(kappa/(1+kappa)) + sqrt(1/(1+kappa))* (((randn(1,ANTENNA) + 1i*randn)/sqrt(2))/(sqrt(d2^path2)));
g1 = sqrt(kappa/(1+kappa)) + sqrt(1/(1+kappa))* (((randn + 1i*randn)/sqrt(2))/(sqrt(d3^path3)));
g2 = sqrt(kappa/(1+kappa)) + sqrt(1/(1+kappa))* (((randn + 1i*randn)/sqrt(2))/(sqrt(d4^path3)));
H1 = h1'*h1;
H2 = h2'*h2;

syms x
eqn = exp(1/x)*expint(1/x) == Rt2/log(2);
V = vpasolve(eqn,x);
Y = double(V);

c1 = Y*SIGMA + SIGMA;
c2 = (norm(g2)/norm(g1))^2*Y*SIGMA + SIGMA;

cvx_begin
variable W(ANTENNA, ANTENNA, 2) hermitian semidefinite
variables alpha1 alpha2
maximize (log(1+alpha1)/log(2) + log(1+alpha2)/log(2))
subject to
    trace(H1*W(:,:,1)) >= alpha1*c1;
    trace(H2*W(:,:,2)) >= (alpha2^2 + trace(H2*W(:,:,1))^2)/2 + alpha2*c2;
    trace(W(:,:,1)) + trace(W(:,:,2)) <= P0;
cvx_end
