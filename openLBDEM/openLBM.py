import numpy as np
import taichi as ti
@ti.data_oriented
class MaskAndGroup:
    def __init__(self,NX,NY):
        self.mask=ti.field(ti.i32,shape=(NX,NY))
        self.group=ti.Vector.field(2,int,shape=(NX*NY,))
        self.count=ti.field(int,shape=())

    @ti.kernel
    def sort(self):
        for i,j in self.mask:
            if self.mask[i,j]==1:
                idx = ti.atomic_add(self.count[None], 1)
                self.group[idx]=ti.Vector([i,j])



@ti.data_oriented
class LBField:
    def __init__(self,name, NX, NY,num_components):
        self.name=name
        self.NX=NX
        self.NY=NY
        self.num_components=ti.field(int,shape=())
        self.num_components[None]=num_components

        # 分布函数（双缓冲）
        self.f = ti.Vector.field(9, float, shape=(num_components,NX, NY)) #populations (old)
        self.f2 = ti.Vector.field(9, float, shape=(num_components,NX, NY)) #populations (new)
        self.SCforce=ti.Vector.field(2, float, shape=(num_components,NX, NY))

        # 宏观量
        self.rho = ti.field(float, shape=(num_components,NX, NY)) #density of components
        self.vel = ti.Vector.field(2, float, shape=(NX, NY)) # fluid velocity 
        self.body_force=ti.Vector.field(2,float,shape=(num_components,NX,NY)) #force populations
        self.total_rho=ti.field(float, shape=(NX, NY)) # density
        self.pressure=ti.field(float, shape=(NX, NY))  #pressure

        self.rho_solid=ti.field(float, shape=(NX, NY)) #use for shan-chen


        # D2Q9模型参数
        self.NPOP = 9 
        self.weights = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0 
        self.c = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        
        #boundary info
        self.gravity_force=ti.Vector([0.0,0.0]) #gravity


        self.neighbor=ti.Matrix.field(n=9,m=2,dtype=int,shape=(self.NX,self.NY))
        self.neighbor_boundary=ti.Vector.field(9,dtype=int,shape=(self.NX,self.NY))

        self.mask=ti.field(int,shape=(self.NX,self.NY)) # 掩码：1-流体, -1-固体, 2-亲水固体, 3-疏水
        
        self.fluid_boundary=MaskAndGroup(self.NX,self.NY)# Fluid=wall+insider
        self.inside_boundary=MaskAndGroup(self.NX,self.NY)
        self.wall_boundary=MaskAndGroup(self.NX,self.NY)#wall=fluid boundary
        self.inlet_boundary=MaskAndGroup(self.NX,self.NY)#wall>inlet
        self.outlet_boundary=MaskAndGroup(self.NX,self.NY)#wall>outlet

        #post
        self.img=ti.Vector.field(3,dtype=ti.f32,shape=(self.NX,self.NY))
        
        self.neighbor_boundary.fill([0,0,0,0,0,0,0,0,0])
        self.mask.fill(1)
        self.SCforce.fill([0.0,0.0])

        #unit conversion
        self.Ct=1.0 #time
        self.Cl=1.0 #length
        self.C_rho=1.0 #density
        self.Cu=1.0 #velocity
        self.C_pressure=1.0
        self.C_force=1.0 # force conversion
        self.C_torque=1.0 

        self.Cnu=1.0 #Kinematic viscosity 
        self.shear_viscosity_LB=ti.field(float, shape=(num_components,)) 
        self.bulk_viscosity_LB=ti.field(float, shape=(num_components,)) 
        
    def init_conversion(self,Cl,Ct,Crho,shear_viscosity,bulk_viscosity):
        self.Ct=Ct
        self.Cl=Cl
        self.C_rho=Crho
        self.Cu=self.Cl/self.Ct
        self.C_force=self.Cl**3*self.C_rho/self.Ct**2 
        self.C_torque=self.C_force*self.Cl
        self.C_pressure=self.C_rho*self.Cu**2
        self.Cnu=self.Cl**2/self.Ct

        for component in range(self.num_components[None]):
            self.shear_viscosity_LB[component]=shear_viscosity[component]/self.Cnu
            self.bulk_viscosity_LB[component]=bulk_viscosity[component]/self.Cnu

    def set_gravity(self,g):
        self.gravity_force=g

    def get_Cforce(self):
        return self.C_force
    
    def get_Ctorque(self):
        return self.C_torque
    
    def get_Cu(self):
        return self.Cu
    

    @ti.kernel 
    def init_hydro_IB(self,sphere_field:ti.template()):
        # 初始化球体区域
        for n in range(sphere_field.num[None]):
            center_x = sphere_field.Sphere[n].pos.x
            center_y = sphere_field.Sphere[n].pos.y
            radius = sphere_field.Sphere[n].radius
            radius_sq = radius ** 2

            # 计算球体覆盖的网格范围
            min_ix = int(ti.max(0, center_x - radius))
            max_ix = int(ti.min(self.NX - 1, center_x + radius))
            min_iy = int(ti.max(0, center_y - radius))
            max_iy = int(ti.min(self.NY - 1, center_y + radius))

            # 遍历球体覆盖的网格区域
            for ix in range(min_ix, max_ix + 1):
                for iy in range(min_iy, max_iy + 1):
                    if (ix - center_x) ** 2 + (iy - center_y) ** 2 <= radius_sq:
                        self.vel[ix,iy]=[.0,.0]
        print("init hydro IB")

    @ti.kernel 
    def init_hydro(self,vel:ti.types.vector(2, ti.f32),pressure_lnlet:float):
        if pressure_lnlet==0.0:
            for m in range(self.fluid_boundary.count[None]):
                ix,iy=self.fluid_boundary.group[m]
                self.vel[ix,iy]=vel
                for component in range(self.num_components[None]):
                    self.rho[component,ix,iy]=1.0/ self.num_components[None]
        else:
            rho_inlet=1+pressure_lnlet*3/self.C_pressure
            for m in range(self.fluid_boundary.count[None]):
                ix,iy=self.fluid_boundary.group[m]
                k=(1.0-rho_inlet)/self.NX
                self.vel[ix,iy]=ti.Vector([.0,.0])
                for component in range(self.num_components[None]):
                    self.rho[component,ix,iy]=(k*ix+rho_inlet)/ self.num_components[None]
        print("init hydro")

    def init_simulation(self,vel,pressure_lnlet,sphere_field=None): 
        self.init_hydro(vel,pressure_lnlet)
        if sphere_field!=None:
            self.init_hydro_IB(sphere_field)
        

    @ti.kernel
    def init_LBM(self,collsion:ti.template()):
        for m in range(self.fluid_boundary.count[None]):
            i,j=self.fluid_boundary.group[m]
            for component in range(self.num_components[None]):
                self.f[component,i,j]=self.f2[component,i,j]=collsion.f_eq(self.c,self.weights,self.vel[i,j],self.rho[component,i,j])
        print("init LBM")





@ti.data_oriented
class MacroscopicEngine:
    @ti.kernel
    def density(self,lb_field:ti.template()):
        lb_field.total_rho.fill(.0)
        lb_field.rho.fill(.0)

        for m in range(lb_field.fluid_boundary.count[None]):
            i,j=lb_field.fluid_boundary.group[m]
            for component in  range(lb_field.num_components[None]):
                # compute the density and uncorrected velocity
                for k in ti.static(range(lb_field.NPOP)):
                    lb_field.rho[component,i,j]+=lb_field.f[component,i,j][k]
                lb_field.total_rho[i,j]+=lb_field.rho[component,i,j]

    @ti.kernel
    def pressure(self,lb_field:ti.template(),sc_filed:ti.template()):
        lb_field.pressure.fill(0.0)
        for m in range(lb_field.fluid_boundary.count[None]):
            i,j=lb_field.fluid_boundary.group[m]
            gas=0.0
            phase=0.0
            for component1 in  range(lb_field.num_components[None]):
                gas+=lb_field.rho[component1,i,j]
                for component2 in  range(lb_field.num_components[None]):
                    phase+=sc_filed.g[component1,component2]*sc_filed.psi(lb_field.rho[component1,i,j]*sc_filed.psi(lb_field.rho[component2,i,j]))
            lb_field.pressure[i,j]=(gas+phase*0.5)/3.0


    @ti.kernel
    def force_density(self,lb_field:ti.template()):
        for m in range(lb_field.fluid_boundary.count[None]):
            i,j=lb_field.fluid_boundary.group[m]
            for component in  range(lb_field.num_components[None]):
                lb_field.body_force[component,i,j]=(lb_field.SCforce[component,i,j]+lb_field.rho[component,i,j]/lb_field.total_rho[i,j]*lb_field.gravity_force)


    @ti.kernel
    def velocity(self,lb_field:ti.template()):
        lb_field.vel.fill([.0,.0])
        for m in range(lb_field.fluid_boundary.count[None]):
            i, j = lb_field.fluid_boundary.group[m]
            for component in  range(lb_field.num_components[None]):
                vel_temp = ti.Vector([0.0, 0.0])
                for k in ti.static(range(lb_field.NPOP)):
                    vel_temp.x += lb_field.f[component,i, j][k] * lb_field.c[k, 0]
                    vel_temp.y += lb_field.f[component,i, j][k] * lb_field.c[k, 1]
                
                vel_temp+=0.5 * lb_field.body_force[component,i, j]
                lb_field.vel[i,j]+=vel_temp
            
            lb_field.vel[i, j] /= lb_field.total_rho[i, j]


    def post_pressure(self,lb_field:ti.template()):
        pressure=lb_field.pressure.to_numpy()
        return pressure

    def post_vel(self,lb_field:ti.template()):
        vel = lb_field.vel.to_numpy()
        vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
        return vel_mag
    
    def writeVTK(self,fname,lb_field:ti.template()):
        rho=lb_field.total_rho.to_numpy().T.flatten()  
        vel=lb_field.vel.to_numpy()
        velx=vel[:,:,0].T.flatten()  
        vely=vel[:,:,1].T.flatten()  

        # bodyforce=lb_field.body_force.to_numpy()
        # bodyforcex=bodyforce[:,:,0].T.flatten()
        # bodyforcey=bodyforce[:,:,1].T.flatten()

        x_coords = np.arange(lb_field.NX)  
        y_coords = np.arange(lb_field.NY)  
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  

        x_flat = x_mesh.flatten()  
        y_flat = y_mesh.flatten()  


        filename = fname + ".vtk"  
        with open(filename, 'w') as fout:  
            fout.write("# vtk DataFile Version 3.0\n")  
            fout.write("Hydrodynamics representation\n")  
            fout.write("ASCII\n\n")  
            fout.write("DATASET STRUCTURED_GRID\n")  
            fout.write(f"DIMENSIONS {lb_field.NX} {lb_field.NY} 1\n")  
            fout.write(f"POINTS {lb_field.NX*lb_field.NY} double\n")  
          
            np.savetxt(fout, np.column_stack((x_flat, y_flat, np.zeros_like(x_flat))), fmt='%.0f')  
          
            fout.write("\n")  
            fout.write(f"POINT_DATA {lb_field.NX*lb_field.NY}\n")  
          
            fout.write("SCALARS Pressure double\n")  
            fout.write("LOOKUP_TABLE Pressure_table\n")  
            np.savetxt(fout, (rho - 1) * lb_field.C_pressure/3.0, fmt='%.8f') 


            fout.write("VECTORS velocity double\n")  
            velocity_data = np.column_stack((velx * lb_field.Cu, vely * lb_field.Cu, np.zeros_like(velx)))  
            np.savetxt(fout, velocity_data, fmt='%.8f') 
  
            # fout.write("VECTORS f double\n")  
            # bodyforce = np.column_stack((bodyforcex, bodyforcey, np.zeros_like(bodyforcex)))  
            # np.savetxt(fout, bodyforce, fmt='%.8f') 

        print(filename)