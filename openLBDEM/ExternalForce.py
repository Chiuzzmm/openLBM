import taichi as ti
import taichi.math as tm

@ti.data_oriented
class Pseudopotential():
    def __init__(self,params: dict):
        default_params={
            'a':None,
            'b':None,
            'R':1.0,
            'G':1.0,
        }
        params = {**default_params, **params}
        self.a=params['a']
        self.b=params['b']
        self.R=params['R']
        self.G=params['G']
        self.rho_cr=params['rho_cr']
        self.pressure_cr=params['pressure_cr']
        self.temperature_cr=params['temperature_cr']
        self.show_info()

    @ti.func
    def EOS(self):
        pass

    @ti.func
    def get_psi(self,dens,T):
        p_EOS=self.EOS(dens,T)
        psi=tm.sqrt(6./self.G*(p_EOS-dens/3.0))
        return psi

    def show_info(self):
        print("="*20)
        print("EOS_pars")
        print(f"  a: {self.a}, b={self.b}")
        print(f"  rho_cr: {self.rho_cr}, p_cr={self.pressure_cr},T_cr={self.temperature_cr}")

@ti.data_oriented
class SC_psi():
    def __init__(self,rho0):
        self.rho0=rho0

    @ti.func
    def get_psi(self,dens,T):
        return self.rho0*(1.0-tm.exp(-dens/self.rho0))


@ti.data_oriented
class vdW_psi(Pseudopotential):
    def __init__(self,params: dict):
        super().__init__(params)

    @ti.func
    def EOS(self,dens,T):
        p=dens*self.R*T/(1.0-self.b*dens)-self.a*dens**2.
        return p
    

@ti.data_oriented
class RK_psi(Pseudopotential):
    def __init__(self,params: dict):
        super().__init__(params)

    @ti.func
    def EOS(self,dens,T):
        p=dens*self.R*T/(1.0-self.b*dens)-self.a*dens**2./tm.sqrt(T)/(1.+self.b*dens)
        return p
    
    
@ti.data_oriented
class PR_psi(Pseudopotential):
    def __init__(self, params:dict):
        super().__init__(params)
        self.omega=params['omega']

    @ti.func
    def EOS(self,dens,T):
        alpha=(1+(0.37464+1.54226*self.omega-0.269922*self.omega**2)*(1-tm.sqrt(T/self.temperature_cr)))**2
        p=dens*self.R*T/(1.0-self.b*dens)-self.a*alpha*dens**2/(1.0+2.0*self.b*dens-(self.b*dens)**2)
        return p


@ti.data_oriented
class CS_psi(Pseudopotential):
    def __init__(self, params:dict):
        super().__init__(params)

    @ti.func
    def EOS(self,dens,T):
        brho4=self.b*dens/4.
        aaa=1.+brho4+brho4**2-brho4**3
        bbb=(1-brho4)**3
        p=dens*self.R*T*aaa/bbb-self.a*dens**2
        return p

@ti.data_oriented
class ShanChenModel:
    def __init__(self,lb_field:ti.template(),group):
        self.group=group
        self.wc = ti.Vector.field(2, float, shape=(9))

        for k in range(9):
            c = ti.Vector([lb_field.c[k, 0], lb_field.c[k, 1]])
            self.wc[k] = lb_field.weights[k] * c


                
@ti.data_oriented
class ShanChenForceC1(ShanChenModel):
    def __init__(self,params: dict):
        default_params={
            'lb_field':None,
            'g_coh':None,
            'group':None,
            'psi':None,
            'fluid_strategy':None
        }
        params = {**default_params, **params}

        super().__init__(params['lb_field'],params['group'])
        self.g_coh = params['g_coh'] #interaction strength
        self.fluid_strategy =params['fluid_strategy']
        self.psi=params['psi']

    @ti.kernel
    def apply(self,lb_field:ti.template()):
        #reset
        lb_field.SCforce.fill([.0,.0])
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            F = ti.Vector([0.0, 0.0])
            for k in ti.static(range(1,9)):
                c=ti.Vector([lb_field.c[k,0],lb_field.c[k,1]])
                x2 = i + c.x
                y2 = j + c.y
                rhoT=self.fluid_strategy.value_fn(x2, y2)
                F+=self.wc[k]*self.psi.get_psi(rhoT.x,rhoT.y)

            F*=(-self.g_coh*self.psi.get_psi(lb_field.rho[i,j,0],lb_field.T[i,j]))

            lb_field.SCforce[i,j,0]+=F



@ti.data_oriented
class ShanChenForceC2(ShanChenModel):
    def __init__(self,params: dict):
        default_params={
            'lb_field':None,
            'g_coh':None,
            'gadh':None,
            'group':None,
            'psi1':None,
            'psi2':None,
            'fluid_strategy':None
        }
        params = {**default_params, **params}

        super().__init__(params['lb_field'],params['group'])
        self.g_coh=params['g_coh']
        self.g_adh=params['gadh']
        self.psi1=params['psi1']
        self.psi2=params['psi2']
        self.fluid_strategy=params['fluid_strategy']

        
    @ti.kernel
    def apply(self,lb_field:ti.template()): # type: ignore
        #reset
        lb_field.SCforce.fill([.0,.0])
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            for component1 in range(lb_field.num_components[None]):
                F_coh = ti.Vector([0.0, 0.0])
                F_adh=ti.Vector([0.0, 0.0])
                for k in ti.static(range(1,9)):
                    c=ti.Vector([lb_field.c[k,0],lb_field.c[k,1]])
                    x2 = i + c.x
                    y2 = j + c.y
                    if self.fluid_strategy.geometry_fn(x2, y2):
                        for component2 in range(lb_field.num_components[None]):
                            rhoT=self.fluid_strategy.value_fn(x2, y2,component2)
                            F_coh+=self.wc[k]*self.psi2.get_psi(rhoT.x,rhoT.y)*self.g_coh[component1,component2]
                    else:
                        # rho_neighbor = lb_field.rho_solid[x2, y2, component1]
                        if lb_field.mask[i,j]!=-1:
                            F_adh+=self.wc[k]*self.g_adh[component1]    
                rho_local = self.psi1.get_psi(lb_field.rho[i, j, component1],lb_field.T[i,j])
                F_total = (-rho_local) * (F_coh + F_adh)
                lb_field.SCforce[i, j, component1] = F_total