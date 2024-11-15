
from itertools import count
from sre_constants import RANGE
from tkinter import SEL, Image
from tkinter.tix import ButtonBox
import matplotlib.pyplot
import taichi as ti
import taichi.math as tm
import sys
import numpy as np
from matplotlib import cm
import math
from PIL import Image



@ti.data_oriented
class LBIBForm:
    def __init__(
        self,
        name,
        NX,  # domain size
        NY
        ):

        #case info
        self.name=name


        #mesh setting
        self.NX=NX
        self.NY=NY

        #LBM setting
        self.NPOP=9
        self.weights = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0 
        self.c = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])

        self.omega_sym=2.0
        self.omega_antisym=1.0


        self.velcity_conversion=1.0
        self.pressure_conversion=1.0
        self.denstiy_conversion=1.0

        self.f = ti.Vector.field(9, float, shape=(NX, NY)) #populations (old)
        self.f2 = ti.Vector.field(9, float, shape=(NX, NY)) #populations (new)
        self.Neighbordata=ti.Matrix.field(n=9,m=2,dtype=int,shape=(NX,NY))
        self.NeighbordataBoundary=ti.Vector.field(9,dtype=int,shape=(NX,NY))

        self.NeighbordataBoundary.fill([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        #phy info
        self.rho = ti.field(float, shape=(NX, NY)) #density
        self.vel = ti.Vector.field(2, float, shape=(NX, NY)) # fluid velocity 
        self.bodyForce=ti.Vector.field(2,float,shape=(NX,NY)) #force populations
        self.gravityForce=ti.Vector([0.0,0.0]) #gravity

        self.vel.fill([0.0,0.0]) 
        self.rho.fill(0.0)
        self.bodyForce.fill([0.0,0.0])

        #boundary info
        self.boundary_condition=1 #1=inlet,2=period,(note period with no obstacle)
        self.InletMode=1 #1= velocity inlet, 2= pressure inlet
        self.rho_Inlet=1.0 #pressure inlet 
        self.vel_wall_Inlet=ti.types.vector(2,float)(0.0,0.0) # velociy inlet

        self.mask=ti.field(int,shape=(NX,NY))
        self.mask.fill(1)

        self.mask_FluidGroup=ti.field(ti.i32,shape=(self.NX,self.NY)) # Fluid=wall+insider
        self.mask_InsideGroup=ti.field(ti.i32,shape=(self.NX,self.NY))
        self.mask_WallGroup=ti.field(ti.i32,shape=(self.NX,self.NY)) #wall=fluid boundary
        self.mask_InletGroup=ti.field(ti.i32,shape=(self.NX,self.NY)) #wall>inlet
        self.mask_OutletGroup=ti.field(ti.i32,shape=(self.NX,self.NY)) #wall>outlet
        self.mask_FluidGroup.fill(0)
        self.mask_InsideGroup.fill(0)
        self.mask_WallGroup.fill(0)
        self.mask_InletGroup.fill(0)
        self.mask_OutletGroup.fill(0)

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

        




    def hydroInfo(self,omega_sym:float,omega_antisym:float):
        self.omega_sym=omega_sym
        self.omega_antisym=omega_antisym

    def boundaryInfo(self,boundary:int=1,InletMode:int=1,vx:float=0.0,vy:float=0.0,p:float=0.0):
        self.boundary_condition=boundary
        self.InletMode=InletMode
        self.vel_wall_Inlet[0]=vx
        self.vel_wall_Inlet[1]=vy
        self.rho_Inlet=p


    def conversion_coefficient(self,Cu_py:float,C_pressure_py:float,C_rho_py:float):
        self.velcity_conversion=Cu_py
        self.pressure_conversion=C_pressure_py
        self.denstiy_conversion=C_rho_py

    @ti.kernel 
    def init_hydro(self):
        count_FluidGroup=self.count_FluidGroup[None]
        if self.InletMode==1:
            #fluild
            for m in range(count_FluidGroup):
                ix,iy=self.boundaryGroup_Fluid[m]
                self.vel[ix,iy]=self.vel_wall_Inlet
                self.rho[ix,iy]=1.0
        else:
            for m in range(count_FluidGroup):
                ix,iy=self.boundaryGroup_Fluid[m]
                k=(1-self.rho_Inlet)/self.NX
                self.vel[ix,iy]=self.vel_wall_Inlet
                self.rho[ix,iy]=k*ix+self.rho_Inlet
        print("init hydyo")


    @ti.kernel
    def init_LBM(self):
        for i,j in ti.ndrange(self.NX, self.NY):
            self.f[i,j]=self.f2[i,j]=self.f_eq(i,j)
        print("init LBM")

    @ti.kernel
    def Mask_rectangle_identify(self,xmin:float,xmax:float,ymin:float,ymax:float):
        for i,j in self.mask:
            if i<xmax and i>xmin:
                if j<ymax and j>ymin:
                    self.mask[i,j]=-1

    @ti.kernel
    def Boundary_identify(self):
        #Group identify
        for ix,iy in self.mask:
            if self.mask[ix,iy]!=-1:
                flag=0
                self.mask_FluidGroup[ix,iy]=1
                self.mask_InsideGroup[ix,iy]=1

                for k in ti.static(range(9)):
                    ix2=ix-self.c[k,0]
                    iy2=iy-self.c[k,1]
                    if ix2<0 or ix2>self.NX-1 or iy2<0 or iy2>self.NY-1 or self.mask[ix2,iy2]==-1:
                        self.Neighbordata[ix,iy][k,0]=-1
                        self.Neighbordata[ix,iy][k,1]=-1
                        flag=1
                    else:
                        self.Neighbordata[ix,iy][k,0]=ix2
                        self.Neighbordata[ix,iy][k,1]=iy2

                if flag==1:
                    self.mask_WallGroup[ix,iy]=1
                    self.mask_InsideGroup[ix,iy]=0

                    if ix==0:
                        self.mask_InletGroup[ix,iy]=1
                    elif ix==self.NX-1:
                        self.mask_OutletGroup[ix,iy]=1

        #boundary  Bounce-Back identify
        for ix,iy in self.mask_WallGroup:
            if self.mask_WallGroup[ix,iy]==1:
                for k in ti.static(range(9)):
                    ix2=self.Neighbordata[ix,iy][k,0]
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
                        self.NeighbordataBoundary[ix,iy][k]=ix2

        # delete max and min in group inlet and outlet
        self.mask_InletGroup[0,0]=0
        self.mask_InletGroup[0,self.NY-1]=0

        self.mask_OutletGroup[self.NX-1,0]=0
        self.mask_OutletGroup[self.NX-1,self.NY-1]=0

        for i,j in self.mask:
            if self.mask_FluidGroup[i,j]==1:
                idx = ti.atomic_add(self.count_FluidGroup[None], 1)
                self.boundaryGroup_Fluid[idx]=ti.Vector([i,j])

            if self.mask_InsideGroup[i,j]==1:
                idx=ti.atomic_add(self.count_InsideGroup[None],1)
                self.boundaryGroup_Inside[idx]=ti.Vector([i,j])

            if self.mask_WallGroup[i,j]==1:
                idx=ti.atomic_add(self.count_WallGroup[None],1)
                self.boundaryGroup_Wall[idx]=ti.Vector([i,j])

            if self.mask_InletGroup[i,j]==1:
                idx=ti.atomic_add(self.count_InletGroup[None],1)
                self.boundaryGroup_Inlet[idx]=ti.Vector([i,j])

            if self.mask_OutletGroup[i,j]==1:
                idx=ti.atomic_add(self.count_OutletGroup[None],1)
                self.boundaryGroup_Outlet[idx]=ti.Vector([i,j])

        print("Boundary identify")


    @ti.kernel
    def computeDensity(self):
        count_FluidGroup=self.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j=self.boundaryGroup_Fluid[m]
            self.rho[i,j]=0.0
            # compute the density and uncorrected velocity
            for k in ti.static(range(9)):
                ti.atomic_add(self.rho[i,j],self.f[i,j][k])


    @ti.kernel
    def computeVelocity(self):
        count_FluidGroup=self.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j=self.boundaryGroup_Fluid[m]
            self.vel[i,j]=ti.Vector([0.0,0.0])
            for k in ti.static(range(9)):
                ti.atomic_add(self.vel[i,j].x,self.f[i,j][k]*self.c[k,0])
                ti.atomic_add(self.vel[i,j].y,self.f[i,j][k]*self.c[k,1])

                ti.atomic_add(self.vel[i,j].x,0.5*self.bodyForce[i,j].x)
                ti.atomic_add(self.vel[i,j].y,0.5*self.bodyForce[i,j].y)
            self.vel[i,j]/=self.rho[i,j]

    @ti.kernel
    def computerForceDensity(self):
        #Reset forces
        self.bodyForce.fill([0.0,0.0])
        count_FluidGroup=self.count_FluidGroup[None]

        for m in range(count_FluidGroup):
            i,j=self.boundaryGroup_Fluid[m]
            ti.atomic_add(self.bodyForce[i,j],self.gravityForce)


    @ti.func#LBM solve
    def f_eq(self,i,j):
        eu=self.c @ self.vel[i,j]
        uv=tm.dot(self.vel[i,j],self.vel[i,j])
        return self.weights*self.rho[i,j]*(1+3*eu+4.5*eu*eu-1.5*uv)

    @ti.func#LBM solve
    def f_force(self,i,j):
        F=self.bodyForce[i,j]
        cF=self.c @ F
        cu=self.c @self.vel[i,j]
        uF=tm.dot(self.vel[i,j],F)
        return self.weights*(3*cF+9*cF*cu-3*uF)

    @ti.kernel#LBM solve
    def collide_bulk(self):
        count_FluidGroup=self.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j = self.boundaryGroup_Fluid[m]

            feqeq=self.f_eq(i, j)
            force_ij=self.f_force(i,j)

            f_sym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            feq_sym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            force_sym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

            f_antisym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            feq_antisym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            force_antisym=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

            #symmetrical and antisymmetrical particle distribution functions
            f_sym[0]=self.f[i,j][0]
            f_sym[1]=(self.f[i,j][1]+self.f[i,j][3])/2
            f_sym[2]=(self.f[i,j][2]+self.f[i,j][4])/2
            f_sym[3]=f_sym[1]
            f_sym[4]=f_sym[2]
            f_sym[5]=(self.f[i,j][5]+self.f[i,j][7])/2
            f_sym[6]=(self.f[i,j][6]+self.f[i,j][8])/2
            f_sym[7]=f_sym[5]
            f_sym[8]=f_sym[6]


            feq_sym[0]=feqeq[0]
            feq_sym[1]=(feqeq[1]+feqeq[3])/2
            feq_sym[2]=(feqeq[2]+feqeq[4])/2
            feq_sym[3]=feq_sym[1]
            feq_sym[4]=feq_sym[2]
            feq_sym[5]=(feqeq[5]+feqeq[7])/2
            feq_sym[6]=(feqeq[6]+feqeq[8])/2
            feq_sym[7]=feq_sym[5]
            feq_sym[8]=feq_sym[6]


            force_sym[0]=force_ij[0]
            force_sym[1]=(force_ij[1]+force_ij[3])/2
            force_sym[2]=(force_ij[2]+force_ij[4])/2
            force_sym[3]=force_sym[1]
            force_sym[4]=force_sym[2]
            force_sym[5]=(force_ij[5]+force_ij[7])/2
            force_sym[6]=(force_ij[6]+force_ij[8])/2
            force_sym[7]=force_sym[5]
            force_sym[8]=force_sym[6]

            for k in range(9):
                f_antisym[k]=self.f[i,j][k]-f_sym[k]
                feq_antisym[k]=feqeq[k]-feq_sym[k]
                force_antisym[k]=force_ij[k]-force_sym[k]

            for k in range(9):
                CollisionOperator =-self.omega_sym * (f_sym[k]-feq_sym[k])-self.omega_antisym*(f_antisym[k]-feq_antisym[k])
                ForceTerm=(1.0-0.5*self.omega_sym)*force_sym[k]+(1.0-0.5*self.omega_antisym)*force_antisym[k]
                self.f2[i,j][k]=self.f[i,j][k]+CollisionOperator+ForceTerm

    @ti.kernel#LBM solve
    def stream(self):

        
        if self.boundary_condition==1:#inlet mode
            #stream in InsideGroup
            count_InsideGroup=self.count_InsideGroup[None]
            for m in range(count_InsideGroup):
                i,j =self.boundaryGroup_Inside[m]
                for k in ti.static(range(9)):
                    ix2=self.Neighbordata[i,j][k,0]
                    iy2=self.Neighbordata[i,j][k,1]
                    self.f[i,j][k]=self.f2[ix2,iy2][k]
        else:#period mode
            for i,j in self.f:
                for k in ti.static(range(9)):
                    ix2=(i+self.c[k,0]+self.NX)%self.NX
                    iy2=(j+self.c[k,1]+self.NY)%self.NY
                    self.f[i,j][k]=self.f2[ix2,iy2][k]

    @ti.func    
    def NEEM(self,ix,iy,ix2,iy2):
        feqeq_b=self.f_eq(ix,iy)
        feqeq_f=self.f_eq(ix2,iy2)
        return feqeq_b+self.f[ix2,iy2]-feqeq_f

    @ti.kernel
    def update_bounce_back (self):
        #inlet mode
        if self.boundary_condition==1:
            count_WallGroup=self.count_WallGroup[None]
            for m in range(count_WallGroup):
                i,j=self.boundaryGroup_Wall[m]
                for k in ti.static(range(9)):
                    ix2=self.Neighbordata[i,j][k,0]
                    iy2=self.Neighbordata[i,j][k,1]
                    if ix2!=-1:
                        #stream in boundary
                        self.f[i,j][k]=self.f2[ix2,iy2][k]
                    else:
                        #bounce back on the static wall 
                        ipop=self.NeighbordataBoundary[i,j][k]
                        self.f[i,j][k]=self.f2[i,j][ipop]

            count_InletGroup=self.count_InletGroup[None]
            if self.InletMode==1:
                #Perform velocity bounce back on the inlet
                for m in range(count_InletGroup):
                    i,j=self.boundaryGroup_Inlet[m]

                    cv=self.c[1,0]*self.vel_wall_Inlet[0]+self.c[1,1]*self.vel_wall_Inlet[1]
                    ti.atomic_add(self.f[i,j][1],6*self.weights[1]*self.rho[i,j]*cv)

                    cv=self.c[5,0]*self.vel_wall_Inlet[0]+self.c[5,1]*self.vel_wall_Inlet[1]
                    ti.atomic_add(self.f[i,j][5],6*self.weights[5]*self.rho[i,j]*cv)

                    cv=self.c[8,0]*self.vel_wall_Inlet[0]+self.c[8,1]*self.vel_wall_Inlet[1]
                    ti.atomic_add(self.f[i,j][8],6*self.weights[8]*self.rho[i,j]*cv)

            else:
                for m in range(count_InletGroup):
                    i,j=self.boundaryGroup_Inlet[m]
                    ix2=self.Neighbordata[i,j][3,0]
                    iy2=self.Neighbordata[i,j][3,1]
                    self.vel[i,j]=self.vel[ix2,iy2]
                    self.rho[i,j]=self.rho_Inlet
                    self.f[i,j]=self.NEEM(i,j,ix2,iy2)

            #Perform Pressure bounce back on the Outlet
            count_OutletGroup=self.count_OutletGroup[None]
            for m in range(count_OutletGroup):
                i,j = self.boundaryGroup_Outlet[m]
                ix2=self.Neighbordata[i,j][1,0]
                iy2=self.Neighbordata[i,j][1,1]
                self.rho[i,j]=1.0
                self.vel[i,j]=self.vel[ix2,iy2]
                self.f[i,j]=self.NEEM(i,j,ix2,iy2)



    @ti.kernel
    def png_cau(self):
        self.img.fill([252/ 255,255/ 255,245/ 255])

        boundaryGroup_Inside=self.count_InsideGroup[None]
        for m in range(boundaryGroup_Inside):
            i,j=self.boundaryGroup_Inside[m]
            self.img[i,j]=ti.Vector([52 / 255,152 / 255,219/ 255])

        boundaryGroup_Wall=self.count_WallGroup[None]
        for m in range(boundaryGroup_Wall):
            i,j=self.boundaryGroup_Wall[m]
            self.img[i, j] = ti.Vector([1.0, 0.0, 0.0])

        boundaryGroup_Inlet=self.count_InletGroup[None]
        for m in range(boundaryGroup_Inlet):
            i,j=self.boundaryGroup_Inlet[m]
            self.img[i,j]=ti.Vector([143/ 255,24/ 255,172/ 255])

        boundaryGroup_Outlet=self.count_OutletGroup[None]
        for m in range(boundaryGroup_Outlet):
            i,j=self.boundaryGroup_Outlet[m]
            self.img[i,j]=ti.Vector([255 / 255,190 / 255,53/ 255])

    def writing_boundary(self):
        self.png_cau()
        img_np=self.img.to_numpy()
        img_pil=Image.fromarray((img_np*255).astype(np.uint8))
        img_pil.save('boundary.png')
        print("writing_boundary")


        
        
    def LBIB_solve(self):
        self.computeDensity()

        self.computerForceDensity()

        self.computeVelocity()
        
        self.collide_bulk()
        
        self.stream()
        
        self.update_bounce_back()


    def post_vel(self):
        vel = self.vel.to_numpy()
        vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5 
        return vel_mag


    def post_pressure(self):
        density=self.rho.to_numpy() 
        pressure=(density-1)
        return pressure


    def writeVTK(self,fname):
        rho=self.rho.to_numpy().flatten()  
        vel=self.vel.to_numpy()
        velx=vel[:,:,0].T.flatten()  
        vely=vel[:,:,1].T.flatten()  

        bodyforce=self.bodyForce.to_numpy()
        bodyforcex=bodyforce[:,:,0].T.flatten()
        bodyforcey=bodyforce[:,:,1].T.flatten()

        x_coords = np.arange(self.NX)  
        y_coords = np.arange(self.NY)  
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)  

        x_flat = x_mesh.flatten()  
        y_flat = y_mesh.flatten()  


        filename = fname + ".vtk"  
        with open(filename, 'w') as fout:  
            fout.write("# vtk DataFile Version 3.0\n")  
            fout.write("Hydrodynamics representation\n")  
            fout.write("ASCII\n\n")  
            fout.write("DATASET STRUCTURED_GRID\n")  
            fout.write(f"DIMENSIONS {self.NX} {self.NY} 1\n")  
            fout.write(f"POINTS {self.NX*self.NY} double\n")  
          
            np.savetxt(fout, np.column_stack((x_flat, y_flat, np.zeros_like(x_flat))), fmt='%.0f')  
          
            fout.write("\n")  
            fout.write(f"POINT_DATA {self.NX*self.NY}\n")  
          
            fout.write("SCALARS Pressure double\n")  
            fout.write("LOOKUP_TABLE Pressure_table\n")  

            np.savetxt(fout, (rho - 1) * self.pressure_conversion, fmt='%.8f') 


            fout.write("VECTORS velocity double\n")  

            velocity_data = np.column_stack((velx * self.velcity_conversion, vely * self.velcity_conversion, np.zeros_like(velx)))  
            np.savetxt(fout, velocity_data, fmt='%.8f') 
  
            fout.write("VECTORS f double\n")  

            bodyforce = np.column_stack((bodyforcex, bodyforcey, np.zeros_like(bodyforcex)))  
            np.savetxt(fout, bodyforce, fmt='%.8f') 

        print(filename)

