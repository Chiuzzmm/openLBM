from tkinter import SEL, Image
import taichi as ti
import taichi.math as tm
import numpy as np
from PIL import Image



@ti.data_oriented
class LBField:
    def __init__(self,name, NX, NY):
        self.name=name
        self.NX=NX
        self.NY=NY

        # 分布函数（双缓冲）
        self.f = ti.Vector.field(9, float, shape=(NX, NY)) #populations (old)
        self.f2 = ti.Vector.field(9, float, shape=(NX, NY)) #populations (new)

        # 宏观量
        self.rho = ti.field(float, shape=(NX, NY)) #density
        self.pressure=ti.field(float, shape=(NX, NY))  #pressure
        self.vel = ti.Vector.field(2, float, shape=(NX, NY)) # fluid velocity 
        self.bodyForce=ti.Vector.field(2,float,shape=(NX,NY)) #force populations
        
        # D2Q9模型参数
        self.NPOP = 9 
        self.weights = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0 
        self.c = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        
        #boundary info
        self.InletMode=1 #1= velocity inlet, 2= pressure inlet
        self.rho_Inlet=1.0 #pressure inlet 
        self.vel_wall_Inlet=ti.Vector([0.0,0.0]) # velociy inlet
        self.gravityForce=ti.Vector([0.0,0.0]) #gravity


        self.Neighbordata=ti.Matrix.field(n=9,m=2,dtype=int,shape=(self.NX,self.NY))
        self.NeighbordataBoundary=ti.Vector.field(9,dtype=int,shape=(self.NX,self.NY))

        self.mask=ti.field(int,shape=(self.NX,self.NY))
        
        self.mask_FluidGroup=ti.field(ti.i32,shape=(self.NX,self.NY)) # Fluid=wall+insider
        self.mask_InsideGroup=ti.field(ti.i32,shape=(self.NX,self.NY))
        self.mask_WallGroup=ti.field(ti.i32,shape=(self.NX,self.NY)) #wall=fluid boundary
        self.mask_InletGroup=ti.field(ti.i32,shape=(self.NX,self.NY)) #wall>inlet
        self.mask_OutletGroup=ti.field(ti.i32,shape=(self.NX,self.NY)) #wall>outlet

        self.count_FluidGroup=ti.field(int,shape=())
        self.count_InsideGroup=ti.field(int,shape=())
        self.count_WallGroup=ti.field(int,shape=())
        self.count_InletGroup=ti.field(int,shape=())
        self.count_OutletGroup=ti.field(int,shape=())

        self.boundaryGroup_Fluid=ti.Vector.field(2,int,shape=(self.NX*self.NY,))
        self.boundaryGroup_Inside=ti.Vector.field(2,int,shape=(self.NX*self.NY,))
        self.boundaryGroup_Wall=ti.Vector.field(2,int,shape=(self.NX*self.NY,))
        self.boundaryGroup_Inlet=ti.Vector.field(2,int,shape=(self.NX*self.NY,))
        self.boundaryGroup_Outlet=ti.Vector.field(2,int,shape=(self.NX*self.NY,))

        #post
        self.img=ti.Vector.field(3,dtype=ti.f32,shape=(self.NX,self.NY))

        self.velcity_conversion=1.0
        self.pressure_conversion=1.0

        self.SCforce=ti.Vector.field(2, float, shape=(NX, NY))
        

        self.NeighbordataBoundary.fill([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        self.mask.fill(1)
        self.mask_FluidGroup.fill(0)
        self.mask_InsideGroup.fill(0)
        self.mask_WallGroup.fill(0)
        self.mask_InletGroup.fill(0)
        self.mask_OutletGroup.fill(0)
        self.SCforce.fill([0.0,0.0])



    def BoundaryInfo(self,InletMode=1,rho_Inlet=1.0,vel_Inlet=ti.Vector([0.0,0.0])):
        self.InletMode=InletMode #1= velocity inlet, 2= pressure inlet
        self.rho_Inlet=rho_Inlet #pressure inlet 
        self.vel_wall_Inlet=vel_Inlet # velociy inlet

    def conversion_coefficient(self,Cu_py:float,C_pressure_py:float):
        self.velcity_conversion=Cu_py
        self.pressure_conversion=C_pressure_py

    def post_pressure(self):
        density=self.rho.to_numpy() 
        pressure=(density-1)
        return pressure

    def post_vel(self):
        vel = self.vel.to_numpy()
        vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
        return vel_mag





@ti.data_oriented
class BoundaryEngine:
    @ti.kernel
    def Boundary_identify(self,lb_filed:ti.template()):
        #Group identify
        for ix,iy in lb_filed.mask:
            if lb_filed.mask[ix,iy]!=-1:
                flag=0
                lb_filed.mask_FluidGroup[ix,iy]=1
                lb_filed.mask_InsideGroup[ix,iy]=1

                for k in ti.static(range(lb_filed.NPOP)):
                    ix2=ix-lb_filed.c[k,0]
                    iy2=iy-lb_filed.c[k,1]
                    if ix2<0 or ix2>lb_filed.NX-1 or iy2<0 or iy2>lb_filed.NY-1 or lb_filed.mask[ix2,iy2]==-1:
                        lb_filed.Neighbordata[ix,iy][k,0]=-1
                        lb_filed.Neighbordata[ix,iy][k,1]=-1
                        flag=1
                    else:
                        lb_filed.Neighbordata[ix,iy][k,0]=ix2
                        lb_filed.Neighbordata[ix,iy][k,1]=iy2

                if flag==1:
                    lb_filed.mask_WallGroup[ix,iy]=1
                    lb_filed.mask_InsideGroup[ix,iy]=0

                    if ix==0:
                        lb_filed.mask_InletGroup[ix,iy]=1
                    elif ix==lb_filed.NX-1:
                        lb_filed.mask_OutletGroup[ix,iy]=1

        #boundary  Bounce-Back identify
        for ix,iy in lb_filed.mask_WallGroup:
            if lb_filed.mask_WallGroup[ix,iy]==1:
                for k in ti.static(range(lb_filed.NPOP)):
                    ix2=lb_filed.Neighbordata[ix,iy][k,0]
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
                        lb_filed.NeighbordataBoundary[ix,iy][k]=ix2

        # delete max and min in group inlet and outlet
        lb_filed.mask_InletGroup[0,0]=0
        lb_filed.mask_InletGroup[0,lb_filed.NY-1]=0

        lb_filed.mask_OutletGroup[lb_filed.NX-1,0]=0
        lb_filed.mask_OutletGroup[lb_filed.NX-1,lb_filed.NY-1]=0

        for i,j in lb_filed.mask:
            if lb_filed.mask_FluidGroup[i,j]==1:
                idx = ti.atomic_add(lb_filed.count_FluidGroup[None], 1)
                lb_filed.boundaryGroup_Fluid[idx]=ti.Vector([i,j])

            if lb_filed.mask_InsideGroup[i,j]==1:
                idx=ti.atomic_add(lb_filed.count_InsideGroup[None],1)
                lb_filed.boundaryGroup_Inside[idx]=ti.Vector([i,j])

            if lb_filed.mask_WallGroup[i,j]==1:
                idx=ti.atomic_add(lb_filed.count_WallGroup[None],1)
                lb_filed.boundaryGroup_Wall[idx]=ti.Vector([i,j])

            if lb_filed.mask_InletGroup[i,j]==1:
                idx=ti.atomic_add(lb_filed.count_InletGroup[None],1)
                lb_filed.boundaryGroup_Inlet[idx]=ti.Vector([i,j])

            if lb_filed.mask_OutletGroup[i,j]==1:
                idx=ti.atomic_add(lb_filed.count_OutletGroup[None],1)
                lb_filed.boundaryGroup_Outlet[idx]=ti.Vector([i,j])

        print("Boundary identify")

    @ti.kernel
    def png_cau(self,lb_filed:ti.template()):
        lb_filed.img.fill([252/ 255,255/ 255,245/ 255])

        boundaryGroup_Inside=lb_filed.count_InsideGroup[None]
        for m in range(boundaryGroup_Inside):
            i,j=lb_filed.boundaryGroup_Inside[m]
            lb_filed.img[i,j]=ti.Vector([52 / 255,152 / 255,219/ 255])

        boundaryGroup_Wall=lb_filed.count_WallGroup[None]
        for m in range(boundaryGroup_Wall):
            i,j=lb_filed.boundaryGroup_Wall[m]
            lb_filed.img[i, j] = ti.Vector([1.0, 0.0, 0.0])

        boundaryGroup_Inlet=lb_filed.count_InletGroup[None]
        for m in range(boundaryGroup_Inlet):
            i,j=lb_filed.boundaryGroup_Inlet[m]
            lb_filed.img[i,j]=ti.Vector([143/ 255,24/ 255,172/ 255])

        boundaryGroup_Outlet=lb_filed.count_OutletGroup[None]
        for m in range(boundaryGroup_Outlet):
            i,j=lb_filed.boundaryGroup_Outlet[m]
            lb_filed.img[i,j]=ti.Vector([255 / 255,190 / 255,53/ 255])

    def writing_boundary(self,lb_filed:ti.template()):
        self.png_cau(lb_filed=lb_filed)

        img_np=lb_filed.img.to_numpy()
        img_pil=Image.fromarray((img_np*255).astype(np.uint8))
        img_pil.save('boundary.png')
        print("writing_boundary")

    @ti.kernel
    def Mask_rectangle_identify(self,lb_filed:ti.template(),xmin:float,xmax:float,ymin:float,ymax:float):
        for i,j in lb_filed.mask:
            if i<xmax and i>xmin:
                if j<ymax and j>ymin:
                    lb_filed.mask[i,j]=-1

    @ti.kernel
    def Mask_cricle_identify(self,lb_filed:ti.template(),x:float,y:float,r:float):
        for ix,iy in lb_filed.mask:
            if (ix-x)**2+(iy-y)**2<r**2:
                lb_filed.mask[ix,iy]=-1

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
    
    @ti.kernel
    def BounceBackInside (self,lb_filed:ti.template()):
        #BB
        count_WallGroup=lb_filed.count_WallGroup[None]
        for m in range(count_WallGroup):
            i,j=lb_filed.boundaryGroup_Wall[m]
            for k in ti.static(range(9)):
                ix2=lb_filed.Neighbordata[i,j][k,0]
                iy2=lb_filed.Neighbordata[i,j][k,1]
                if ix2!=-1:
                    #stream in boundary
                    lb_filed.f[i,j][k]=lb_filed.f2[ix2,iy2][k]
                else:
                    #bounce back on the static wall 
                    ipop=lb_filed.NeighbordataBoundary[i,j][k]
                    lb_filed.f[i,j][k]=lb_filed.f2[i,j][ipop]

    @ti.kernel
    def BounceBackInlet(self,lb_filed:ti.template()):
        #inlet
        count_InletGroup=lb_filed.count_InletGroup[None]
        if lb_filed.InletMode==1:
            #Perform velocity bounce back on the inlet
            for m in range(count_InletGroup):
                i,j=lb_filed.boundaryGroup_Inlet[m]

                cv=lb_filed.c[1,0]*lb_filed.vel_wall_Inlet[0]+lb_filed.c[1,1]*lb_filed.vel_wall_Inlet[1]
                ti.atomic_add(lb_filed.f[i,j][1],6*lb_filed.weights[1]*lb_filed.rho[i,j]*cv)

                cv=lb_filed.c[5,0]*lb_filed.vel_wall_Inlet[0]+lb_filed.c[5,1]*lb_filed.vel_wall_Inlet[1]
                ti.atomic_add(lb_filed.f[i,j][5],6*lb_filed.weights[5]*lb_filed.rho[i,j]*cv)

                cv=lb_filed.c[8,0]*lb_filed.vel_wall_Inlet[0]+lb_filed.c[8,1]*lb_filed.vel_wall_Inlet[1]
                ti.atomic_add(lb_filed.f[i,j][8],6*lb_filed.weights[8]*lb_filed.rho[i,j]*cv)
        else:
            for m in range(count_InletGroup):
                i,j=lb_filed.boundaryGroup_Inlet[m]
                ix2=lb_filed.Neighbordata[i,j][3,0]
                iy2=lb_filed.Neighbordata[i,j][3,1]
                lb_filed.vel[i,j]=lb_filed.vel[ix2,iy2]
                lb_filed.rho[i,j]=lb_filed.rho_Inlet
                lb_filed.f[i,j]=self.NEEM(lb_filed.c,lb_filed.weights,lb_filed.vel[i,j],lb_filed.rho[i,j],lb_filed.vel[ix2,iy2],lb_filed.rho[ix2,iy2],lb_filed.f[ix2,iy2])

                #ABC
                # uw=1.5*lb_filed.vel[i,j]-0.5*lb_filed.vel[ix2,iy2]
                # lb_filed.vel[i,j]=uw
                # lb_filed.rho[i,j]=lb_filed.rho_Inlet
                # ftemp=self.ABC(lb_filed.c,lb_filed.weights,lb_filed.f[i,j],lb_filed.rho_Inlet,uw)
                # lb_filed.f[i,j][1]=ftemp[1]
                # lb_filed.f[i,j][5]=ftemp[5]
                # lb_filed.f[i,j][8]=ftemp[8]

    @ti.kernel
    def BounceBackOutlet(self,lb_filed:ti.template()):
        #Perform Pressure bounce back on the Outlet
        count_OutletGroup=lb_filed.count_OutletGroup[None]
        for m in range(count_OutletGroup):

            i,j = lb_filed.boundaryGroup_Outlet[m]
            ix2=lb_filed.Neighbordata[i,j][1,0]
            iy2=lb_filed.Neighbordata[i,j][1,1]

            #NEEM
            lb_filed.rho[i,j]=1.0
            lb_filed.vel[i,j]=lb_filed.vel[ix2,iy2]
            lb_filed.f[i,j]=self.NEEM(lb_filed.c,lb_filed.weights,lb_filed.vel[i,j],lb_filed.rho[i,j],lb_filed.vel[ix2,iy2],lb_filed.rho[ix2,iy2],lb_filed.f[ix2,iy2])

            #ABC
            # uw=1.5*lb_filed.vel[i,j]-0.5*lb_filed.vel[ix2,iy2]
            # lb_filed.vel[i,j]=uw
            # lb_filed.rho[i,j]=1.0
            # ftemp=self.ABC(lb_filed.c,lb_filed.weights,lb_filed.f[i,j],1.0,uw)
            # lb_filed.f[i,j][3]=ftemp[3]
            # lb_filed.f[i,j][6]=ftemp[6]
            # lb_filed.f[i,j][7]=ftemp[7]

@ti.data_oriented
class GlobalEngine:
    @ti.kernel 
    def init_hydro(self,lb_filed:ti.template()):
        count_FluidGroup=lb_filed.count_FluidGroup[None]
        if lb_filed.InletMode==1:
            for m in range(count_FluidGroup):
                ix,iy=lb_filed.boundaryGroup_Fluid[m]
                lb_filed.vel[ix,iy]=lb_filed.vel_wall_Inlet
                lb_filed.rho[ix,iy]=1.0
        else:
            for m in range(count_FluidGroup):
                ix,iy=lb_filed.boundaryGroup_Fluid[m]
                k=(1.0-lb_filed.rho_Inlet)/lb_filed.NX
                lb_filed.vel[ix,iy]=lb_filed.vel_wall_Inlet
                lb_filed.rho[ix,iy]=k*ix+lb_filed.rho_Inlet
        print("init hydyo")


    @ti.kernel
    def init_LBM(self,lb_filed:ti.template(),collsion:ti.template()):
        for i,j in ti.ndrange(lb_filed.NX, lb_filed.NY):
            lb_filed.f[i,j]=lb_filed.f2[i,j]=collsion.f_eq(lb_filed.c,lb_filed.weights,lb_filed.vel[i,j],lb_filed.rho[i,j])
        print("init LBM")


    def writeVTK(self,fname,lb_filed:ti.template()):
        rho=lb_filed.rho.to_numpy().T.flatten()  
        vel=lb_filed.vel.to_numpy()
        velx=vel[:,:,0].T.flatten()  
        vely=vel[:,:,1].T.flatten()  

        bodyforce=lb_filed.bodyForce.to_numpy()
        bodyforcex=bodyforce[:,:,0].T.flatten()
        bodyforcey=bodyforce[:,:,1].T.flatten()

        x_coords = np.arange(lb_filed.NX)  
        y_coords = np.arange(lb_filed.NY)  
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  

        x_flat = x_mesh.flatten()  
        y_flat = y_mesh.flatten()  


        filename = fname + ".vtk"  
        with open(filename, 'w') as fout:  
            fout.write("# vtk DataFile Version 3.0\n")  
            fout.write("Hydrodynamics representation\n")  
            fout.write("ASCII\n\n")  
            fout.write("DATASET STRUCTURED_GRID\n")  
            fout.write(f"DIMENSIONS {lb_filed.NX} {lb_filed.NY} 1\n")  
            fout.write(f"POINTS {lb_filed.NX*lb_filed.NY} double\n")  
          
            np.savetxt(fout, np.column_stack((x_flat, y_flat, np.zeros_like(x_flat))), fmt='%.0f')  
          
            fout.write("\n")  
            fout.write(f"POINT_DATA {lb_filed.NX*lb_filed.NY}\n")  
          
            fout.write("SCALARS Pressure double\n")  
            fout.write("LOOKUP_TABLE Pressure_table\n")  
            np.savetxt(fout, (rho - 1) * lb_filed.pressure_conversion/3.0, fmt='%.8f') 


            fout.write("VECTORS velocity double\n")  
            velocity_data = np.column_stack((velx * lb_filed.velcity_conversion, vely * lb_filed.velcity_conversion, np.zeros_like(velx)))  
            np.savetxt(fout, velocity_data, fmt='%.8f') 
  
            fout.write("VECTORS f double\n")  
            bodyforce = np.column_stack((bodyforcex, bodyforcey, np.zeros_like(bodyforcex)))  
            np.savetxt(fout, bodyforce, fmt='%.8f') 

        print(filename)




@ti.data_oriented
class MacroscopicEngine:
    @ti.kernel
    def computeDensity(self,lb_filed:ti.template()):
        count_FluidGroup=lb_filed.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j=lb_filed.boundaryGroup_Fluid[m]
            lb_filed.rho[i,j]=0.0
            # compute the density and uncorrected velocity
            for k in ti.static(range(lb_filed.NPOP)):
                lb_filed.rho[i,j]+=lb_filed.f[i,j][k]
            
    @ti.kernel
    def computePressure(self,lb_filed:ti.template(),sc_filed:ti.template()):
        lb_filed.pressure.fill(0.0)
        count_FluidGroup = lb_filed.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i, j = lb_filed.boundaryGroup_Fluid[m]
            lb_filed.pressure[i,j]=lb_filed.rho[i,j]/3.0+sc_filed.gA/6.0*lb_filed.psi(lb_filed.rho[i,j])**2

    @ti.kernel
    def computerForceDensity(self,lb_filed:ti.template()):
        #Reset forces
        lb_filed.bodyForce.fill([0.0,0.0])
        count_FluidGroup=lb_filed.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j=lb_filed.boundaryGroup_Fluid[m]
            ti.atomic_add(lb_filed.bodyForce[i,j],lb_filed.gravityForce+lb_filed.SCforce[i,j])

    @ti.kernel
    def computeVelocity(self,lb_filed:ti.template()):
        count_FluidGroup = lb_filed.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i, j = lb_filed.boundaryGroup_Fluid[m]
            lb_filed.vel[i, j] = ti.Vector([0.0, 0.0])
            
            vel_temp = ti.Vector([0.0, 0.0])
            for k in ti.static(range(lb_filed.NPOP)):
                vel_temp.x += lb_filed.f[i, j][k] * lb_filed.c[k, 0]
                vel_temp.y += lb_filed.f[i, j][k] * lb_filed.c[k, 1]
            
            vel_temp+=0.5 * lb_filed.bodyForce[i, j]

            lb_filed.vel[i, j]+=vel_temp
            lb_filed.vel[i, j] /= lb_filed.rho[i, j]

@ti.data_oriented
class CollisionEngine:
    @ti.func
    def f_eq(self,c,w,vel,rho):
        eu=c @ vel
        uv=tm.dot(vel,vel)
        return w*rho*(1+3*eu+4.5*eu*eu-1.5*uv)

    @ti.func
    def f_force(self,c,w,F,vel):
        cF=c @ F
        cu=c @vel
        uF=tm.dot(vel,F)
        return w*(3*cF+9*cF*cu-3*uF)
    

@ti.data_oriented
class BGKCollision(CollisionEngine):
    def __init__(self):
        super().__init__()
        self.omega = 1.0 

    def Relaxation_pars(self,omega):
        self.omega=omega

    @ti.kernel#LBM solve
    def Collision(self,lb_filed:ti.template()):
        count_FluidGroup=lb_filed.count_FluidGroup[None]

        for m in range(count_FluidGroup):
            i,j = lb_filed.boundaryGroup_Fluid[m]

            feqeq=self.f_eq(lb_filed.c,lb_filed.weights,lb_filed.vel[i,j],lb_filed.rho[i,j])
            force_ij=self.f_force(lb_filed.c,lb_filed.weights,lb_filed.bodyForce[i,j],lb_filed.vel[i,j])

            CollisionOperator =-self.omega*(lb_filed.f[i,j]-feqeq)
            ForceTerm=(1.0-0.5*self.omega)*force_ij

            lb_filed.f2[i,j]=lb_filed.f[i,j]+CollisionOperator+ForceTerm

@ti.data_oriented
class TRTCollision(CollisionEngine):
    def __init__(self):
        super().__init__()
        self.omega_sym=2.0
        self.omega_antisym=1.0
    
    def Relaxation_pars(self,omega_sym,omega_antisym):
        self.omega_sym=omega_sym
        self.omega_antisym=omega_antisym

    @ti.func
    def f_sym(self,f):
        f_sym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        f_sym[0]=f[0]
        f_sym[1]=(f[1]+f[3])/2
        f_sym[2]=(f[2]+f[4])/2
        f_sym[3]=f_sym[1]
        f_sym[4]=f_sym[2]
        f_sym[5]=(f[5]+f[7])/2
        f_sym[6]=(f[6]+f[8])/2
        f_sym[7]=f_sym[5]
        f_sym[8]=f_sym[6]
        return f_sym
    
    @ti.func
    def feq_sym(self,feqeq):
        feq_sym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        feq_sym[0]=feqeq[0]
        feq_sym[1]=(feqeq[1]+feqeq[3])/2
        feq_sym[2]=(feqeq[2]+feqeq[4])/2
        feq_sym[3]=feq_sym[1]
        feq_sym[4]=feq_sym[2]
        feq_sym[5]=(feqeq[5]+feqeq[7])/2
        feq_sym[6]=(feqeq[6]+feqeq[8])/2
        feq_sym[7]=feq_sym[5]
        feq_sym[8]=feq_sym[6]
        return feq_sym
    
    @ti.func
    def force_sym(self,force):
        force_sym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        force_sym[0]=force[0]
        force_sym[1]=(force[1]+force[3])/2
        force_sym[2]=(force[2]+force[4])/2
        force_sym[3]=force_sym[1]
        force_sym[4]=force_sym[2]
        force_sym[5]=(force[5]+force[7])/2
        force_sym[6]=(force[6]+force[8])/2
        force_sym[7]=force_sym[5]
        force_sym[8]=force_sym[6]
        return force_sym
    
    @ti.kernel#LBM solve
    def Collision(self,lb_filed:ti.template()):
        count_FluidGroup=lb_filed.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j = lb_filed.boundaryGroup_Fluid[m]

            feqeq=self.f_eq(lb_filed.c,lb_filed.weights,lb_filed.vel[i,j],lb_filed.rho[i,j])
            force_ij=self.f_force(lb_filed.c,lb_filed.weights,lb_filed.bodyForce[i,j],lb_filed.vel[i,j])

            #symmetrical and antisymmetrical particle distribution functions
            f_sym=self.f_sym(lb_filed.f[i,j])
            feq_sym=self.feq_sym(feqeq)
            force_sym=self.force_sym(force_ij)

            f_antisym=lb_filed.f[i,j]-f_sym
            feq_antisym=feqeq-feq_sym
            force_antisym=force_ij-force_sym

            CollisionOperator =-self.omega_sym * (f_sym-feq_sym)-self.omega_antisym*(f_antisym-feq_antisym)
            ForceTerm=(1.0-0.5*self.omega_sym)*force_sym+(1.0-0.5*self.omega_antisym)*force_antisym
            lb_filed.f2[i,j]=lb_filed.f[i,j]+CollisionOperator+ForceTerm

@ti.data_oriented
class MRTCollision(CollisionEngine):
    def __init__(self):
        super().__init__()
        self.M=ti.field(float,shape=(9,9))
        self.M_inverse=ti.field(float,shape=(9,9))
        M_np =np.array([
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0],
                [4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
                [0.0, -2.0, 0.0, 2.0, 0.0, -1.0, -1.0,-1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [0.0, 0.0, -2.0, 0.0, 2.0, 1.0, 1.0, -1.0, -1.0],
                [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0]])
        M_inv_np=np.linalg.inv(M_np)
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                self.M[i,j]=M_np[i,j]
                self.M_inverse[i,j]=M_inv_np[i,j]
        self.diag=ti.field(float,shape=(9,))

    def Relaxation_pars(self,omega_e,omega_v,omega_q=1,omega_epsilon=1):
        # self.diag.fill(1.0)
        # self.diag[1]=omega_e
        # self.diag[7]=omega_v
        # self.diag[8]=omega_v
        # self.diag[4]=omega_q
        # self.diag[6]=omega_q
        # self.diag[2]=omega_epsilon

            # book
        # self.diag.fill(0.0)
        # self.diag[1]=omega_e
        # self.diag[2]=omega_epsilon
        # self.diag[4]=omega_q
        # self.diag[6]=omega_q
        # self.diag[7]=omega_v
        # self.diag[8]=omega_v

            #zhihu
        # self.diag.fill(0.0)
        # self.diag[7]=self.diag[8]=omega_v
        # self.diag[1]=self.diag[2]=omega_v
        # self.diag[4]=self.diag[6]=8.0*(2.0-omega_v)/(8.0-omega_v)

            # JR
        self.diag.fill(1.0)
        self.diag[1]=omega_e
        self.diag[7]=omega_v
        self.diag[8]=omega_v

    @ti.func
    def m_eq(self,vel,rho):
        jx=vel.x
        jy=vel.y

        meq=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        meq[0]=1.0
        meq[1]=-2.0+3.0*(jx**2+jy**2)
        meq[2]=1.0-3.0*(jx**2+jy**2)
        meq[3]=jx
        meq[4]=-jx
        meq[5]=jy
        meq[6]=-jy
        meq[7]=(jx**2-jy**2)
        meq[8]=jx*jy

        meq*=rho
        return meq
    

    @ti.func
    def m_force(self,F,vel):
        mforce=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        Fu=tm.dot(F,vel)

        mforce[1]=6*Fu
        mforce[2]=-6*Fu
        mforce[3]=F[0]
        mforce[4]=-F[0]
        mforce[5]=F[1]
        mforce[6]=-F[1]
        mforce[7]=2*(F[0]*vel.x-F[1]*vel.y)
        mforce[8]=F[1]*vel.x+F[0]*vel.y

        return mforce
    
    @ti.kernel#LBM solve
    def Collision(self,lb_filed:ti.template()):
        count_FluidGroup=lb_filed.count_FluidGroup[None]
        for id in range(count_FluidGroup):
            i,j = lb_filed.boundaryGroup_Fluid[id]

            m=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            a=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            b=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            fpop2=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

            #Transform to moment space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    m[ii]+=self.M[ii,jj]*lb_filed.f[i,j][jj]

            meq=self.m_eq(lb_filed.vel[i,j],lb_filed.rho[i,j]) #Compute equilibrium moments
            mf=self.m_force(lb_filed.bodyForce[i,j],lb_filed.vel[i,j]) #Guo Forcing

            for ii in ti.static(range(9)):
                a[ii]=self.diag[ii]*(m[ii]-meq[ii])
                b[ii]=(1.0-self.diag[ii]/2.0)*(mf[ii])
            
            m2=m-a+b#Collide

            #Transform to population space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    ti.atomic_add(fpop2[ii],self.M_inverse[ii,jj]*m2[jj])

            lb_filed.f2[i,j]=fpop2

@ti.data_oriented
class HuangMRTCollision(MRTCollision):
    def __init__(self):
        super().__init__()
        self.k1=1.0
        self.k2=1.0
        self.gA=-5.0
        self.rho0=1.0 
        self.rho_cr=tm.log(2.0)
        self.rho_liq=1.95
        self.rho_gas=0.15
        self.solidCof=0.5
        self.rho_solid=self.rho_gas+self.solidCof*(self.rho_liq-self.rho_gas)

    def ShanChenSetting(self,rho_liq=1.95,rho_gas=0.15,rho_solid_cof=0.9,gA=-5.0):
        self.rho_liq=rho_liq
        self.rho_gas=rho_gas
        self.rho_solid=self.rho_gas+rho_solid_cof*(self.rho_liq-self.rho_gas)
        self.gA=gA

    @ti.func
    def psi(self,dens):
        return self.rho0*(1.0-tm.exp(-dens/self.rho0))

    @ti.func
    def m_Q(self,F,rho):
        Qm=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        F2=tm.dot(F,F)
        a=3*(self.k1+2*self.k2)*F2
        b=self.gA*self.psi(rho)**2
        
        Qm[1]=a/b
        Qm[2]=-Qm[1]
        Qm[7]=self.k1*(F[0]**2-F[1]**2)/b
        Qm[8]=self.k1*F[0]*F[1]/b

        return Qm
    
    @ti.kernel
    def Collision(self,lb_filed:ti.template()):
        count_FluidGroup=lb_filed.count_FluidGroup[None]
        for id in range(count_FluidGroup):
            i,j = lb_filed.boundaryGroup_Fluid[id]

            m=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            a=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            b=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            c=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            fpop2=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

            #Transform to moment space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    ti.atomic_add(m[ii],self.M[ii,jj]*lb_filed.f[i,j][jj])

            meq=self.m_eq(lb_filed.vel[i,j],lb_filed.rho[i,j]) #Compute equilibrium moments
            mf=self.m_force(lb_filed.bodyForce[i,j],lb_filed.vel[i,j]) #Guo Forcing
            mQ=self.m_Q(lb_filed.bodyForce[i,j],lb_filed.rho[i,j]) # huang

            for ii in ti.static(range(9)):
                a[ii]=self.diag[ii]*(m[ii]-meq[ii])
                b[ii]=(1.0-self.diag[ii]/2.0)*(mf[ii])
                c[ii]=self.diag[ii]*mQ[ii]

            m2=m-a+b+c#Collide

            #Transform to population space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    ti.atomic_add(fpop2[ii],self.M_inverse[ii,jj]*m2[jj])

            lb_filed.f2[i,j]=fpop2

    @ti.kernel
    def computeSCForces(self,lb_filed:ti.template()):
        count_FluidGroup=lb_filed.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j = lb_filed.boundaryGroup_Fluid[m]

            lb_filed.SCforce[i,j].x=0.0
            lb_filed.SCforce[i,j].y=0.0

            f_temp_x=0.0
            f_temp_y=0.0
            psinb=0.0
            for k in ti.static(range(1,9)):
                # x2=i+lb_filed.c[k,0]
                # y2=j+lb_filed.c[k,1]
                x2=(i+lb_filed.c[k,0]+lb_filed.NX)%lb_filed.NX 
                y2=(j+lb_filed.c[k,1]+lb_filed.NY)%lb_filed.NY

                if lb_filed.mask[x2,y2]==-1: #solid
                    # if lb_filed.rho[i,j]>lb_filed.rho_liq/8.0:
                    #     psinb=lb_filed.psi(lb_filed.rho_solid)
                    # else:
                    psinb=lb_filed.psi(lb_filed.rho_solid)
                else:
                    psinb=lb_filed.psi(lb_filed.rho[x2,y2])

                f_temp_x+=lb_filed.weights[k]*lb_filed.c[k,0]*psinb
                f_temp_y+=lb_filed.weights[k]*lb_filed.c[k,1]*psinb

            psiloc=lb_filed.psi(lb_filed.rho[i,j])
            f_temp_x*=(-lb_filed.gA*psiloc)
            f_temp_y*=(-lb_filed.gA*psiloc)

            ti.atomic_add(lb_filed.SCforce[i,j].x,f_temp_x)
            ti.atomic_add(lb_filed.SCforce[i,j].y,f_temp_y)

@ti.data_oriented
class StreamEngine:
    @ti.kernel
    def StreamInside(self,lb_filed:ti.template()):
        #stream in InsideGroup
        count_InsideGroup=lb_filed.count_InsideGroup[None]
        for m in range(count_InsideGroup):
            i,j =lb_filed.boundaryGroup_Inside[m]
            for k in ti.static(range(lb_filed.NPOP)):
                ix2=lb_filed.Neighbordata[i,j][k,0]
                iy2=lb_filed.Neighbordata[i,j][k,1]
                lb_filed.f[i,j][k]=lb_filed.f2[ix2,iy2][k]


    @ti.kernel
    def StreamPeriodic(self,lb_filed:ti.template()):
        count_FluidGroup=lb_filed.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j = lb_filed.boundaryGroup_Fluid[m]
            for k in ti.static(range(lb_filed.NPOP)):
                x2=(i-lb_filed.c[k,0]+lb_filed.NX)%lb_filed.NX 
                y2=(j-lb_filed.c[k,1]+lb_filed.NY)%lb_filed.NY
                lb_filed.f[i,j][k]=lb_filed.f2[x2,y2][k]


