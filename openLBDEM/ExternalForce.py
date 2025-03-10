import taichi as ti
import taichi.math as tm


@ti.data_oriented
class ShanChenModel:
    def __init__(self,num_components,group,density_provider):
        self.num_components = num_components
        self.rho0 = 1.0
        self.group=group
        self.denstiy_provider =ti.func(density_provider)

    @ti.func
    def psi(self,dens):
        return self.rho0*(1.0-tm.exp(-dens/self.rho0))

                
@ti.data_oriented
class ShanChenForceC1(ShanChenModel):
    def __init__(self, lb_field:ti.template(),g,group,density_provider):
        super().__init__(lb_field.num_components[None],group,density_provider)
        self.g = g #interaction strength
        

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
                rho_neighbor=self.denstiy_provider(0, x2, y2)

                F+=lb_field.weights[k]*c*self.psi(rho_neighbor)
            F*=(-self.g*self.psi(lb_field.rho[0,i,j]))

            lb_field.SCforce[0,i,j]+=F



@ti.data_oriented
class ShanChenForceC2(ShanChenModel):
    def __init__(self, lb_field:ti.template(),g,group,density_provider):
        super().__init__(lb_field.num_components[None],group,density_provider)
        self.g = ti.field(float,shape=(self.num_components,self.num_components)) #interaction strength
        for component1 in range(self.num_components):
            for component2 in range(self.num_components):
                self.g[component1,component2]=g[component1,component2]
                        
    @ti.kernel
    def apply(self,lb_field:ti.template()): # type: ignore
        #reset
        lb_field.SCforce.fill([.0,.0])
        for m in range(self.group.count[None]):
            i,j = self.group.group[m]
            for component1 in range(self.num_components):
                for component2 in range(self.num_components):
                    F = ti.Vector([0.0, 0.0])
                    rho_neighbor=.0
                    for k in ti.static(range(1,9)):
                        c=ti.Vector([lb_field.c[k,0],lb_field.c[k,1]])
                        x2 = i + c.x
                        y2 = j + c.y
                        rho_neighbor=self.denstiy_provider(component2, x2, y2)

                        F+=lb_field.weights[k]*c*self.psi(rho_neighbor)
                    F*=(-self.g[component1,component2]*self.psi(lb_field.rho[component1,i,j]))

                    lb_field.SCforce[component1,i,j]+=F

