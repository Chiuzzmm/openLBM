function [p] = SCEOS(rho,G)
rho0=1;
cs2=1/3;
p = cs2*rho + cs2*G/2*(rho0*(1-exp(-rho/rho0))).^2;
end