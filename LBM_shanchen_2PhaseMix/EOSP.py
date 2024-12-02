import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def Maxwell(coeff):
    a=3/49
    b=2/21
    _R=1
    _omega=0.344
    _Tc=0.0778/0.45724*a/b/_R
    _pc = 0.0778**2/0.45724*a/b**2
    _T_m = coeff * _Tc
    phi = lambda T : (1 + (0.37464+1.54226*_omega-0.26992*_omega**2)*(1-np.sqrt(T/_Tc)))**2
    P_R_EOS=lambda v,T: (_R*T)/(v-b)-a*phi(T)/(v**2+2*b*v-b**2)
    v_c = opt.root_scalar(lambda v: P_R_EOS(v, _Tc)- _pc, method='brentq', bracket=[1.1*b, 
20000]).root
    
    diff_P_R_EOS = lambda v, T : -_R*T/(v-b)**2 + 2*a*phi(T)*(v+b)/(v**2+2*b*v-b**2)**2
    # 求出diff_P_R_EOS等于0的v
    v_left = opt.root_scalar(lambda v: diff_P_R_EOS(v, _T_m), method='brentq', bracket=[1.1*b, 
    v_c]).root
    v_right = opt.root_scalar(lambda v: diff_P_R_EOS(v, _T_m), method='brentq', bracket=[v_c, 
    20000]).root
    P_left = P_R_EOS(v_left, _T_m)
    P_right = P_R_EOS(v_right, _T_m)
                            
    def fun_area(_T_m, p_Max):
        v_liq = opt.root_scalar(lambda v: P_R_EOS(v, _T_m) - p_Max, method='brentq', bracket=
        [1.1*b, v_left]).root
        v_gas = opt.root_scalar(lambda v: P_R_EOS(v, _T_m) - p_Max, method='brentq', bracket=
        [v_right, 20000]).root
        F = lambda upsilon: _R*_T_m*np.log(abs(upsilon-b))- \
            a*phi(_T_m)/(2*2**0.5*b)*np.log(abs((upsilon+b-2**0.5*b)/(upsilon+b+2**0.5*b)))
        area = F(v_gas) - F(v_liq) - p_Max*(v_gas - v_liq)
        return area
    
    p_Max = opt.root_scalar(lambda p_Max: fun_area(_T_m, p_Max), method='brentq', \
        bracket=[max(P_left*1.000,P_right*0.0005),P_right]).root
    v_liq = opt.root_scalar(lambda v: P_R_EOS(v, _T_m) - p_Max, method='brentq', bracket=[1.1*b,v_left]).root
    v_gas = opt.root_scalar(lambda v: P_R_EOS(v, _T_m) - p_Max, method='brentq', bracket=[v_right, 20000]).root
    return 1/v_liq, 1/v_gas, p_Max

rho_liq = []
rho_gas = []
p_Max = []
for coeff in np.arange(0.5, 0.999, 0.001):
    rho_liq_, rho_gas_, p_Max_ = Maxwell(coeff)
    rho_liq.append(rho_liq_)
    rho_gas.append(rho_gas_)
    p_Max.append(p_Max_)
    
plt.figure(figsize=(12, 5))
# 左图绘制密度，右图绘制压强
plt.subplot(1, 2, 1)
plt.plot(rho_liq, np.arange(0.5, 0.999, 0.001), label='liquid density')
plt.plot(rho_gas, np.arange(0.5, 0.999, 0.001), label='gas density')
plt.legend()
plt.xlabel('density')
plt.ylabel('T/Tc')
# plt.xscale('log')
plt.subplot(1, 2, 2)
plt.plot(p_Max, np.arange(0.5, 0.999, 0.001), label='Maxwell pressure')
plt.legend()
plt.xlabel('pressure')
plt.ylabel('T/Tc')
# plt.xscale('log')
plt.savefig('fig3.png')
plt.tight_layout()
plt.show()