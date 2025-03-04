import taichi as ti
import taichi.math as tm

@ti.data_oriented
class ShanChenForce:
    def __init__(self, lb_field:ti.template()):
        self.g = ti.field(float,shape=(lb_field.num_components[None],lb_field.num_components[None])) #interaction strength
        self.rho0 = 1.0
        self.rho_cr=tm.log(2.0)
        self.rho0_liq=1.93244248895799
        self.rho0_gas=0.156413030238316
        self.rho1_liq=1.76775878467311
        self.rho1_gas=0.187191599699299
        self.solidCof=0.5

    def setting_pars(self,lb_field:ti.template(),g):
        for component1 in range(lb_field.num_components[None]):
                for component2 in range(lb_field.num_components[None]):
                     self.g[component1,component2]=g[component1,component2]
                     

    @ti.func
    def psi(self,dens):
        return self.rho0*(1.0-tm.exp(-dens/self.rho0))

    @ti.kernel
    def apply(self,lb_field:ti.template()):
        #reset
        lb_field.SCforce.fill([.0,.0])

        for m in range(lb_field.fluid_boundary.count[None]):
            i,j = lb_field.fluid_boundary.group[m]
            for component1 in range(lb_field.num_components[None]):
                for component2 in range(lb_field.num_components[None]):
                    F = ti.Vector([0.0, 0.0])
                    rho_neighbor=.0
                    for k in ti.static(range(1,9)):
                        c=ti.Vector([lb_field.c[k,0],lb_field.c[k,1]])
                        x2=(i+lb_field.c[k,0]+lb_field.NX)%lb_field.NX 
                        y2=(j+lb_field.c[k,1]+lb_field.NY)%lb_field.NY
                        if lb_field.mask[x2,y2]==1: #fuild
                            rho_neighbor=lb_field.rho[component2,x2,y2]
                        else: #another
                            rho_neighbor=lb_field.rho_solid[x2,y2]

                        F+=lb_field.weights[k]*c*self.psi(rho_neighbor)
                    F*=(-self.g[component1,component2]*self.psi(lb_field.rho[component1,i,j]))

                    lb_field.SCforce[component1,i,j]+=F

    @ti.kernel
    def init_hydro(self,lb_field:ti.template()):
        lb_field.vel.fill([.0,.0])

        for m in range(lb_field.fluid_boundary.count[None]):
                ix,iy=lb_field.fluid_boundary.group[m]
                # if iy<lb_field.NY/2:
                #     lb_field.rho[0,ix,iy]=self.rho_gas
                #     lb_field.rho[1,ix,iy]=self.rho_liq 
                # else:
                #     lb_field.rho[0,ix,iy]=self.rho_liq 
                #     lb_field.rho[1,ix,iy]=self.rho_gas

                # if (ix-lb_field.NX/2)**2+(iy-lb_field.NY/2)**2<=100*2:
                #     lb_field.rho[0,ix,iy]=self.rho0_gas
                #     lb_field.rho[1,ix,iy]=self.rho0_liq
                # else:
                #     lb_field.rho[0,ix,iy]=self.rho1_gas
                #     lb_field.rho[1,ix,iy]=self.rho1_liq

                lb_field.rho[0,ix,iy]=self.rho_cr+0.1*ti.random()
                lb_field.rho[1,ix,iy]=self.rho1_gas