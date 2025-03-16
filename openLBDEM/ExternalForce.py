import taichi as ti
import taichi.math as tm


@ti.data_oriented
class ShanChenModel:
    def __init__(self,lb_field:ti.template(),group):
        self.num_components = lb_field.num_components[None]
        self.rho0 = 1.0
        self.group=group
        self.wc = ti.Vector.field(2, float, shape=(9))

        for k in range(9):
            c = ti.Vector([lb_field.c[k, 0], lb_field.c[k, 1]])
            self.wc[k] = lb_field.weights[k] * c

    @ti.func
    def psi(self,dens):
        return self.rho0*(1.0-tm.exp(-dens/self.rho0))

                
@ti.data_oriented
class ShanChenForceC1(ShanChenModel):
    def __init__(self, lb_field:ti.template(),g,group,density_provider):
        super().__init__(lb_field,group)
        self.g = g #interaction strength
        self.denstiy_provider =ti.func(density_provider)

    @ti.kernel
    def apply(self,lb_field:ti.template()):
        #reset
        lb_field.SCforce.fill([.0,.0])

        for m in range(self.group.count[None]):
            i,j = self.group.group[m]

            F = ti.Vector([0.0, 0.0])
            rho_neighbor=.0
            for k in ti.static(range(1,9)):
                c=ti.Vector([lb_field.c[k,0],lb_field.c[k,1]])
                x2 = i + c.x
                y2 = j + c.y
                rho_neighbor=self.denstiy_provider(x2, y2,0)

                F+=self.wc[k]*self.psi(rho_neighbor)
            F*=(-self.g*self.psi(lb_field.rho[i,j,0]))

            lb_field.SCforce[i,j,0]+=F



@ti.data_oriented
class ShanChenForceC2(ShanChenModel):
    def __init__(self, lb_field:ti.template(),g_coh,gadh,group,fluid_strategy,solid_strategy=None):
        super().__init__(lb_field,group)
        self.g_coh=ti.field(float,shape=(self.num_components,self.num_components))
        self.g_adh=ti.field(float,shape=(self.num_components))
        self.fluid_strategy=fluid_strategy
        self.solid_strategy=solid_strategy


        for component1 in range(self.num_components):
            self.g_adh[component1]=gadh[component1]
            for component2 in range(self.num_components):
                self.g_coh[component1,component2]=g_coh[component1,component2]
        
        
    @ti.kernel
    def apply(self,lb_field:ti.template()): # type: ignore
        #reset
        lb_field.SCforce.fill([.0,.0])
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            for component1 in range(self.num_components):
                F_coh = ti.Vector([0.0, 0.0])
                F_adh=ti.Vector([0.0, 0.0])
                for k in ti.static(range(1,9)):
                    c=ti.Vector([lb_field.c[k,0],lb_field.c[k,1]])
                    x2 = i + c.x
                    y2 = j + c.y
                    if self.fluid_strategy.geometry_fn(x2, y2):
                        for component2 in range(self.num_components):
                            rho_neighbor = self.fluid_strategy.value_fn(x2, y2, component2)
                            F_coh+=self.wc[k]*self.psi(rho_neighbor)*self.g_coh[component1,component2]
                    else:
                        # rho_neighbor = lb_field.rho_solid[x2, y2, component1]
                        F_adh+=self.wc[k]*self.g_adh[component1]
                rho_local = self.psi(lb_field.rho[i, j, component1])
                F_total = (-rho_local) * (F_coh + F_adh)
                lb_field.SCforce[i, j, component1] = F_total