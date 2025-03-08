from PIL import Image
import taichi as ti
import taichi.math as tm
import numpy as np

@ti.data_oriented
class BoundaryCondition:
    @ti.kernel
    def apply(self):
        pass

    @ti.func
    def f_eq(self,c,w,vel,rho):
        eu=c @ vel
        uv=tm.dot(vel,vel)
        return w*rho*(1+3*eu+4.5*eu*eu-1.5*uv)
    
    @ti.func    
    def NEEM(self,c,w,vel1,rho1,vel2,rho2,f2):
        feqeq_b=self.f_eq(c,w,vel1,rho1)
        feqeq_f=self.f_eq(c,w,vel2,rho2)
        return feqeq_b+f2-feqeq_f

    @ti.func
    def ABC(self,c,w,f,rho_w,uw):
        cu=c@uw
        uw2=tm.dot(uw,uw)
        return  -f+2*w*rho_w*(1.0+4.5*cu*cu-1.5*uw2)

@ti.data_oriented
class WettingBoundary(BoundaryCondition):
    def __init__(self,boundary_group:ti.template(),rho_solid_cof:float,rho_solid:float):
        super().__init__()
        self.boundary_group=boundary_group
        self.rho_solid=rho_solid
    
    @ti.kernel
    def apply(self,lb_field:ti.template()):
        for m in range(self.boundary_group.count[None]):
            i,j=self.boundary_group.group[m]
            lb_field.rho_solid[i,j]=self.rho_solid
    

@ti.data_oriented
class VelocityInlet(BoundaryCondition):
    def __init__(self,boundary_group:ti.template(),velocity): # type: ignore
        super().__init__()
        self.boundary_group=boundary_group
        self.velocity_in=velocity


    @ti.kernel
    def apply(self, lb_field:ti.template()):
        #Perform velocity bounce back on the inlet
        for m in range(self.boundary_group.count[None]):
            i,j=self.boundary_group.group[m]
            for component in range(lb_field.num_components[None]):
                cv=lb_field.c[1,0]*self.velocity_in.x+lb_field.c[1,1]*self.velocity_in.y
                lb_field.f[component,i,j][1]+=6*lb_field.weights[1]*lb_field.rho[component,i,j]*cv

                cv=lb_field.c[5,0]*self.velocity_in.x+lb_field.c[5,1]*self.velocity_in.y
                lb_field.f[component,i,j][5]+=6*lb_field.weights[5]*lb_field.rho[component,i,j]*cv

                cv=lb_field.c[8,0]*self.velocity_in.x+lb_field.c[8,1]*self.velocity_in.y
                lb_field.f[component,i,j][8]+=6*lb_field.weights[8]*lb_field.rho[component,i,j]*cv

@ti.data_oriented
class PressureInlet(BoundaryCondition):
    def __init__(self,boundary_group:ti.template(),rho_in): # type: ignore
        super().__init__()
        self.boundary_group=boundary_group
        self.rho_in=rho_in

    @ti.kernel
    def apply(self, lb_field:ti.template()):
        for m in range(self.boundary_group.count[None]):
            i,j=self.boundary_group.group[m]
            ix2=lb_field.neighbor[i,j][3,0]
            iy2=lb_field.neighbor[i,j][3,1]
            for component in range(lb_field.num_components[None]):
                # lb_field.vel[i,j]=lb_field.vel[ix2,iy2]
                # lb_field.rho[i,j]=self.rho_in
                # lb_field.f[i,j]=self.NEEM(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j],lb_field.vel[ix2,iy2],lb_field.rho[ix2,iy2],lb_field.f[ix2,iy2])

                #ABC
                uw=1.5*lb_field.vel[i,j]-0.5*lb_field.vel[ix2,iy2]
                lb_field.vel[i,j]=uw
                lb_field.total_rho[i,j]=self.rho_in
                ftemp=self.ABC(lb_field.c,lb_field.weights,lb_field.f[component,i,j],self.rho_in,uw)
                lb_field.f[component,i,j][1]=ftemp[1]
                lb_field.f[component,i,j][5]=ftemp[5]
                lb_field.f[component,i,j][8]=ftemp[8]

@ti.data_oriented
class PressureOutlet(BoundaryCondition):
    def __init__(self,boundary_group:ti.template(),rho_out): # type: ignore
        super().__init__()
        self.boundary_group=boundary_group
        self.rho_out=rho_out

    @ti.kernel
    def apply(self, lb_field:ti.template()):
        for m in range(self.boundary_group.count[None]):
            i,j=self.boundary_group.group[m]
            ix2=lb_field.neighbor[i,j][1,0]
            iy2=lb_field.neighbor[i,j][1,1]

            for component in range(lb_field.num_components[None]):
                #NEEM
                # lb_field.rho[i,j]=self.rho_out
                # lb_field.vel[i,j]=lb_field.vel[ix2,iy2]
                # lb_field.f[i,j]=self.NEEM(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j],lb_field.vel[ix2,iy2],lb_field.rho[ix2,iy2],lb_field.f[ix2,iy2])

                #ABC
                uw=1.5*lb_field.vel[i,j]-0.5*lb_field.vel[ix2,iy2]
                lb_field.vel[i,j]=uw
                lb_field.total_rho[i,j]=self.rho_out
                ftemp=self.ABC(lb_field.c,lb_field.weights,lb_field.f[component,i,j],self.rho_out,uw)
                lb_field.f[component,i,j][3]=ftemp[3]
                lb_field.f[component,i,j][6]=ftemp[6]
                lb_field.f[component,i,j][7]=ftemp[7]

@ti.data_oriented
class BounceBackWall(BoundaryCondition):
    def __init__(self,boundary_group:ti.template()):
        super().__init__()
        self.boundary_group=boundary_group

    @ti.kernel
    def apply(self, lb_field:ti.template()): # type: ignore
        for m in range(self.boundary_group.count[None]):
            i,j=self.boundary_group.group[m]
            for component in range(lb_field.num_components[None]):
                
                for k in ti.static(range(9)):
                    ix2=lb_field.neighbor[i,j][k,0]
                    iy2=lb_field.neighbor[i,j][k,1]
                    if ix2!=-1:
                        #stream in boundary
                        lb_field.f[component,i,j][k]=lb_field.f2[component,ix2,iy2][k]
                    else:
                        #bounce back on the static wall 
                        ipop=lb_field.neighbor_boundary[i,j][k]
                        lb_field.f[component,i,j][k]=lb_field.f2[component,i,j][ipop]

@ti.data_oriented
class PeriodicBoundary(BoundaryCondition):
    def __init__(self,axis):
        super().__init__()
        self.axis=axis

    @ti.kernel
    def apply(self, lb_field:ti.template()):
        pass


@ti.data_oriented
class BoundaryEngine:
    def __init__(self):
        self.boundary_conditions = []

    
    def boundary_identify(self,lb_field:ti.template()):
        self.boundary_classify(lb_field=lb_field)
        lb_field.fluid_boundary.sort()
        lb_field.inside_boundary.sort()
        lb_field.wall_boundary.sort()
        lb_field.inlet_boundary.sort()
        lb_field.outlet_boundary.sort()

        print("="*20)
        print("Boundary identify")
        print(f"  fluid_boundary size: {lb_field.fluid_boundary.count}")
        print(f"  inside_boundary size: {lb_field.inside_boundary.count}")
        print(f"  wall_boundary size: {lb_field.wall_boundary.count}")
        print(f"  inlet_boundary size: {lb_field.inlet_boundary.count}")
        print(f"  outlet_boundary size: {lb_field.outlet_boundary.count}")


    @ti.kernel
    def boundary_classify(self,lb_field:ti.template()):
        #group identify
        for ix,iy in lb_field.mask:
            if lb_field.mask[ix,iy]!=-1:
                flag=0
                lb_field.fluid_boundary.mask[ix,iy]=1
                lb_field.inside_boundary.mask[ix,iy]=1

                for k in ti.static(range(lb_field.NPOP)):
                    ix2=ix-lb_field.c[k,0]
                    iy2=iy-lb_field.c[k,1]
                    if ix2<0 or ix2>lb_field.NX-1 or iy2<0 or iy2>lb_field.NY-1 or lb_field.mask[ix2,iy2]==-1:
                        lb_field.neighbor[ix,iy][k,0]=-1
                        lb_field.neighbor[ix,iy][k,1]=-1
                        flag=1
                    else:
                        lb_field.neighbor[ix,iy][k,0]=ix2
                        lb_field.neighbor[ix,iy][k,1]=iy2

                if flag==1:
                    lb_field.wall_boundary.mask[ix,iy]=1
                    lb_field.inside_boundary.mask[ix,iy]=0

                    if ix==0:
                        lb_field.inlet_boundary.mask[ix,iy]=1
                    elif ix==lb_field.NX-1:
                        lb_field.outlet_boundary.mask[ix,iy]=1

        #boundary  Bounce-Back identify
        for ix,iy in lb_field.wall_boundary.mask:
            if lb_field.wall_boundary.mask[ix,iy]==1:
                for k in ti.static(range(lb_field.NPOP)):
                    ix2=lb_field.neighbor[ix,iy][k,0]
                    if ix2==-1:
                        if k == 1:  
                            ix2 = 3  
                        elif k == 2:  
                            ix2 = 4  
                        elif k == 3:  
                            ix2 = 1  
                        elif k == 4:  
                            ix2 = 2  
                        elif k == 5:  
                            ix2 = 7  
                        elif k == 6:  
                            ix2 = 8  
                        elif k == 7:  
                            ix2 = 5  
                        elif k == 8:  
                            ix2 = 6 
                        lb_field.neighbor_boundary[ix,iy][k]=ix2

        # delete max and min in group inlet and outlet
        lb_field.inlet_boundary.mask[0,0]=0
        lb_field.inlet_boundary.mask[0,lb_field.NY-1]=0

        lb_field.outlet_boundary.mask[lb_field.NX-1,0]=0
        lb_field.outlet_boundary.mask[lb_field.NX-1,lb_field.NY-1]=0

    
    
        

    @ti.kernel
    def png_cau(self,lb_field:ti.template()):
        lb_field.img.fill([252/ 255,255/ 255,245/ 255])

        for m in range(lb_field.inside_boundary.count[None]):
            i,j=lb_field.inside_boundary.group[m]
            lb_field.img[i,j]=ti.Vector([52 / 255,152 / 255,219/ 255])

        for m in range(lb_field.wall_boundary.count[None]):
            i,j=lb_field.wall_boundary.group[m]
            lb_field.img[i, j] = ti.Vector([1.0, 0.0, 0.0])

        for m in range(lb_field.inlet_boundary.count[None]):
            i,j=lb_field.inlet_boundary.group[m]
            lb_field.img[i,j]=ti.Vector([143/ 255,24/ 255,172/ 255])

        for m in range(lb_field.outlet_boundary.count[None]):
            i,j=lb_field.outlet_boundary.group[m]
            lb_field.img[i,j]=ti.Vector([255 / 255,190 / 255,53/ 255])

    def writing_boundary(self,lb_field:ti.template()):
        self.png_cau(lb_field=lb_field)

        img_np=lb_field.img.to_numpy()
        img_pil=Image.fromarray((img_np*255).astype(np.uint8))
        img_pil.save('boundary.png')
        print("writing boundary finish")

    @ti.kernel
    def Mask_rectangle_identify(self,lb_field:ti.template(),xmin:float,xmax:float,ymin:float,ymax:float):
        for i,j in lb_field.mask:
            if i<xmax and i>xmin:
                if j<ymax and j>ymin:
                    lb_field.mask[i,j]=-1

    @ti.kernel
    def Mask_cricle_identify(self,lb_field:ti.template(),x:float,y:float,r:float):
        for ix,iy in lb_field.mask:
            if (ix-x)**2+(iy-y)**2<r**2:
                lb_field.mask[ix,iy]=-1
    
    def add_boundary_condition(self,bc):
        self.boundary_conditions.append(bc)
    
    def apply_boundary_conditions(self,lb_field:ti.template()):
        for bc in self.boundary_conditions:
            bc.apply(lb_field)