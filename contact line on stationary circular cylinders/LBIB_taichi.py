
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
        self.M=ti.field(float,shape=(9,9))
        self.M_inverse=ti.field(float,shape=(9,9))
        self.diag=ti.field(float,shape=(9,))

        self.velcity_conversion=1.0
        self.pressure_conversion=1.0

        self.f = ti.Vector.field(9, float, shape=(NX, NY)) #populations (old)
        self.f2 = ti.Vector.field(9, float, shape=(NX, NY)) #populations (new)
        self.Neighbordata=ti.Matrix.field(n=9,m=2,dtype=int,shape=(NX,NY))
        self.NeighbordataBoundary=ti.Vector.field(9,dtype=int,shape=(NX,NY))
        self.NeighbordataBoundary.fill([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        #phy info
        self.rho = ti.field(float, shape=(NX, NY)) #density
        self.pressure=ti.field(float, shape=(NX, NY)) #pressure
        self.vel = ti.Vector.field(2, float, shape=(NX, NY)) # fluid velocity 
        self.bodyForce=ti.Vector.field(2,float,shape=(NX,NY)) #force populations
        self.gravityForce=ti.Vector([0.0,0.0]) #gravity

        # self.gravityForce.x=0.00001

        self.vel.fill([0.0,0.0]) 
        self.rho.fill(0.0)
        self.bodyForce.fill([0.0,0.0])

        #boundary info
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

        #IB info
        self.MDFIteration=1

        #sc 
        self.rho_cr=tm.log(2.0)
        self.rho_liq=1.932442488957992
        self.rho_gas=0.156413030238316
        self.rho0=1.0
        self.rho_solid=self.rho_gas+0.9*(self.rho_liq-self.rho_gas)
 

        self.gA=-5.0
        self.SCforce=ti.Vector.field(2, float, shape=(NX, NY))
        self.SCforce.fill([0.0,0.0])

        #mrt - huang pars
        self.k1=1.0
        self.k2=1.0
        self.epslion=1.0


    @ti.kernel
    def mask_balls(self):
        count_FluidGroup=self.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            ix,iy=self.boundaryGroup_Fluid[m]
            #ball
            for n in range(self.particles_Number):
                center_x=self.particles_pos[n].x
                center_y=self.particles_pos[n].y
                r2=self.particles_radius[n]**2
                if (ix-center_x)**2+(iy-center_y)**2<=r2:
                    self.mask[ix,iy]=-1

        


    def boundaryInfo(self,InletMode:int=1,vx:float=0.0,vy:float=0.0,p:float=0.0):
        self.InletMode=InletMode
        self.vel_wall_Inlet[0]=vx
        self.vel_wall_Inlet[1]=vy
        self.rho_Inlet=p

    def conversion_coefficient(self,Cu_py:float,C_pressure_py:float):
        self.velcity_conversion=Cu_py
        self.pressure_conversion=C_pressure_py


    @ti.kernel 
    def init_hydro(self):
        count_FluidGroup=self.count_FluidGroup[None]
        #fluild
        for m in range(count_FluidGroup):
            ix,iy=self.boundaryGroup_Fluid[m]
            self.vel[ix,iy]=0.0
            
            #bubble
            if iy<self.NY/2:
                self.rho[ix,iy]=self.rho_liq
            else:
                self.rho[ix,iy]=self.rho_gas



        print("init hydyo")

    @ti.kernel 
    def init_MRT(self,omega_e:float,omega_v:float,omega_q:float,omega_epsilon:float):
        #init M
        for k in ti.static(range(9)):
            u2=self.c[k,0]**2+self.c[k,1]**2

            self.M[0,k]=1.0
            self.M[1,k]=-4.0+3.0*u2
            self.M[2,k]=4.0-21.0/2.0*u2+9.0/2.0*u2**2
            self.M[3,k]=self.c[k,0]
            self.M[4,k]=self.c[k,0]*(-5.0+3.0*u2)
            self.M[5,k]=self.c[k,1]
            self.M[6,k]=self.c[k,1]*(-5.0+3.0*u2)
            self.M[7,k]=self.c[k,0]**2-self.c[k,1]**2
            self.M[8,k]=self.c[k,0]*self.c[k,1]

        # init Minveerse
        self.M_inverse.fill(0.0)

        for k in ti.static(range(9)):
            self.M_inverse[k,0]=1/9.0

        self.M_inverse[0,1]=-1/9.0
        self.M_inverse[1,1]=-1/36.0
        self.M_inverse[2,1]=-1/36.0
        self.M_inverse[3,1]=-1/36.0
        self.M_inverse[4,1]=-1/36.0
        self.M_inverse[5,1]=1/18.0
        self.M_inverse[6,1]=1/18.0
        self.M_inverse[7,1]=1/18.0
        self.M_inverse[8,1]=1/18.0

        self.M_inverse[0,2]=1/9.0
        self.M_inverse[1,2]=-1/18.0
        self.M_inverse[2,2]=-1/18.0
        self.M_inverse[3,2]=-1/18.0
        self.M_inverse[4,2]=-1/18.0
        self.M_inverse[5,2]=1/36.0
        self.M_inverse[6,2]=1/36.0
        self.M_inverse[7,2]=1/36.0
        self.M_inverse[8,2]=1/36.0

        self.M_inverse[0,3]=0.0
        self.M_inverse[1,3]=1/6.0
        self.M_inverse[2,3]=0.0
        self.M_inverse[3,3]=-1/6.0
        self.M_inverse[4,3]=0.0
        self.M_inverse[5,3]=1/6.0
        self.M_inverse[6,3]=-1/6.0
        self.M_inverse[7,3]=-1/6.0
        self.M_inverse[8,3]=1/6.0

        self.M_inverse[0,4]=0.0
        self.M_inverse[1,4]=-1/6.0
        self.M_inverse[2,4]=0.0
        self.M_inverse[3,4]=1/6.0
        self.M_inverse[4,4]=0.0
        self.M_inverse[5,4]=1/12.0
        self.M_inverse[6,4]=-1/12.0
        self.M_inverse[7,4]=-1/12.0
        self.M_inverse[8,4]=1/12.0

        self.M_inverse[0,5]=0.0
        self.M_inverse[1,5]=0.0
        self.M_inverse[2,5]=1/6.0
        self.M_inverse[3,5]=0.0
        self.M_inverse[4,5]=-1/6.0
        self.M_inverse[5,5]=1/6.0
        self.M_inverse[6,5]=1/6.0
        self.M_inverse[7,5]=-1/6.0
        self.M_inverse[8,5]=-1/6.0

        self.M_inverse[0,6]=0.0
        self.M_inverse[1,6]=0.0
        self.M_inverse[2,6]=-1/6.0
        self.M_inverse[3,6]=0.0
        self.M_inverse[4,6]=1/6.0
        self.M_inverse[5,6]=1/12.0
        self.M_inverse[6,6]=1/12.0
        self.M_inverse[7,6]=-1/12.0
        self.M_inverse[8,6]=-1/12.0

        self.M_inverse[0,7]=0.0
        self.M_inverse[1,7]=1/4.0
        self.M_inverse[2,7]=-1/4.0
        self.M_inverse[3,7]=1/4.0
        self.M_inverse[4,7]=-1/4.0
        self.M_inverse[5,7]=0.0
        self.M_inverse[6,7]=0.0
        self.M_inverse[7,7]=0.0
        self.M_inverse[8,7]=0.0

        self.M_inverse[5,8]=1/4.0
        self.M_inverse[6,8]=-1/4.0
        self.M_inverse[7,8]=1/4.0
        self.M_inverse[8,8]=-1/4.0
  
        
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
    
    def init_SC(self,k1,epslion):
        self.k1=k1
        self.epslion=epslion
        self.k2=-epslion/8.0-k1


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
    def Mask_cricle_identify(self,x:float,y:float,r:float):
        for ix,iy in self.mask:
            if (ix-x)**2+(iy-y)**2<r**2:
                self.mask[ix,iy]=-1

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
                    # if ix2<0 or ix2>self.NX-1 or iy2<0 or iy2>self.NY-1 or self.mask[ix2,iy2]==-1:
                    if self.mask[ix2,iy2]==-1:
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
                self.rho[i,j]+=self.f[i,j][k]
            
    @ti.kernel
    def computePressure(self):
        self.pressure.fill(0.0)
        count_FluidGroup = self.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i, j = self.boundaryGroup_Fluid[m]
            self.pressure[i,j]=self.rho[i,j]/3.0+self.gA/6.0*self.psi(self.rho[i,j])**2

    @ti.kernel
    def computeVelocity(self):
        count_FluidGroup = self.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i, j = self.boundaryGroup_Fluid[m]
            self.vel[i, j] = ti.Vector([0.0, 0.0])
            
            vel_temp = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                vel_temp.x += self.f[i, j][k] * self.c[k, 0]
                vel_temp.y += self.f[i, j][k] * self.c[k, 1]
            
            vel_temp+=0.5 * self.bodyForce[i, j]

            self.vel[i, j]+=vel_temp
            self.vel[i, j] /= self.rho[i, j]


    @ti.func
    def psi(self,dens):
        return self.rho0*(1.0-tm.exp(-dens/self.rho0))
    
    @ti.kernel
    def computeSCForces(self):
        # fluid
        count_FluidGroup=self.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j=self.boundaryGroup_Fluid[m]
            self.SCforce[i,j].x=0.0
            self.SCforce[i,j].y=0.0

            f_temp_x=0.0
            f_temp_y=0.0
            psinb=0.0


            for k in ti.static(range(1,9)):
                x2=i+self.c[k,0]
                y2=j+self.c[k,1]

                if self.mask[x2,y2]==-1: #solid
                    if self.rho[i,j]>self.rho_liq/8.0:
                        psinb=self.psi(self.rho_solid)
                    else:
                        psinb=self.psi(self.rho_gas)
                elif y2<0 :
                    psinb=self.psi(self.rho_liq)
                elif  y2>self.NY-1:
                    psinb=self.psi(self.rho_gas)
                elif x2<0 or x2>self.NX-1:
                    x2=(i+self.c[k,0]+self.NX)%self.NX 
                    psinb=self.psi(self.rho[x2,y2])
                else:
                    psinb=self.psi(self.rho[x2,y2])
                
                f_temp_x+=self.weights[k]*self.c[k,0]*psinb
                f_temp_y+=self.weights[k]*self.c[k,1]*psinb

            psiloc=self.psi(self.rho[i,j])
            f_temp_x*=(-self.gA*psiloc)
            f_temp_y*=(-self.gA*psiloc)

            self.SCforce[i,j].x+=f_temp_x
            self.SCforce[i,j].y+=f_temp_y


    @ti.kernel
    def computerForceDensity(self):
        count_FluidGroup=self.count_FluidGroup[None]
        for m in range(count_FluidGroup):
            i,j=self.boundaryGroup_Fluid[m]
            self.bodyForce[i,j]=ti.Vector([0.0,0.0]) #reset
            force=(self.gravityForce+self.SCforce[i,j])
            self.bodyForce[i,j]+=force
            

    @ti.func#LBM solve
    def f_eq(self,i,j):
        eu=self.c @ self.vel[i,j]
        uv=tm.dot(self.vel[i,j],self.vel[i,j])
        return self.weights*self.rho[i,j]*(1+3*eu+4.5*eu*eu-1.5*uv)

    
    @ti.func
    def m_eq(self,i,j):
        jx=self.vel[i,j].x
        jy=self.vel[i,j].y
        j2=jx**2+jy**2
        meq=ti.Vector([1.0,-2.0+3.0*j2,1.0-3.0*j2,jx,-jx,jy,-jy,(jx**2-jy**2),jx*jy])
        meq*=self.rho[i,j]
        return meq


    @ti.func#LBM solve
    def f_force(self,i,j):
        F=self.bodyForce[i,j]
        cF=self.c @ F
        cu=self.c @self.vel[i,j]
        uF=tm.dot(self.vel[i,j],F)
        return self.weights*(3*cF+9*cF*cu-3*uF)

    @ti.func
    def m_force(self,i,j):
        mforce=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        F=self.bodyForce[i,j]
        Fu=tm.dot(F,self.vel[i,j])

        mforce[1]=6*Fu
        mforce[2]=-mforce[1]
        mforce[3]=F[0]
        mforce[4]=-F[0]
        mforce[5]=F[1]
        mforce[6]=-F[1]
        mforce[7]=2*(F[0]*self.vel[i,j].x-F[1]*self.vel[i,j].y)
        mforce[8]=F[1]*self.vel[i,j].x+F[0]*self.vel[i,j].y

        return mforce
    
    
    @ti.func
    def m_Q(self,i,j):
        Qm=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        F=self.bodyForce[i,j]
        F2=tm.dot(F,F)
        a=3*(self.k1+2*self.k2)*F2
        b=self.gA*self.psi(self.rho[i,j])**2
        
        Qm[1]=a/b
        Qm[2]=-Qm[1]
        Qm[7]=self.k1*(F[0]**2-F[1]**2)/b
        Qm[8]=self.k1*F[0]*F[1]/b

        return Qm
    
    @ti.kernel#LBM solve
    def collide_bulk(self):
        count_FluidGroup=self.count_FluidGroup[None]
        for id in range(count_FluidGroup):
            i,j = self.boundaryGroup_Fluid[id]

            m=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            a=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            b=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            c=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            fpop2=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

            #Transform to moment space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    ti.atomic_add(m[ii],self.M[ii,jj]*self.f[i,j][jj])

            meq=self.m_eq(i,j) #Compute equilibrium moments
            mf=self.m_force(i,j) #Guo Forcing
            mQ=self.m_Q(i,j) # huang

            for ii in ti.static(range(9)):
                a[ii]=self.diag[ii]*(m[ii]-meq[ii])
                b[ii]=(1.0-self.diag[ii]/2.0)*(mf[ii])
                c[ii]=self.diag[ii]*mQ[ii]

            m2=m-a+b+c#Collide

            #Transform to population space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    ti.atomic_add(fpop2[ii],self.M_inverse[ii,jj]*m2[jj])

            self.f2[i,j]=fpop2


    @ti.kernel#LBM solve
    def stream(self):
        #stream in InsideGroup
        count_InsideGroup=self.count_InsideGroup[None]
        for m in range(count_InsideGroup):
            i,j =self.boundaryGroup_Inside[m]
            for k in ti.static(range(9)):
                ix2=self.Neighbordata[i,j][k,0]
                iy2=self.Neighbordata[i,j][k,1]
                self.f[i,j][k]=self.f2[ix2,iy2][k]

    @ti.func    
    def NEEM(self,ix,iy,ix2,iy2):
        feqeq_b=self.f_eq(ix,iy)
        feqeq_f=self.f_eq(ix2,iy2)
        return feqeq_b+self.f[ix2,iy2]-feqeq_f


    @ti.func
    def ABC(self,ix,iy,rho_w,uw):
        cu=self.c@uw
        uw2=tm.dot(uw,uw)
        return  -self.f[ix,iy]+2*self.weights*rho_w*(1.0+4.5*cu*cu-1.5*uw2)


    @ti.kernel
    def update_bounce_back (self):

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

        #BB
        for i in range(self.NX):
            self.f[i,0][2]=self.f2[i,0][4]
            self.f[i,0][5]=self.f2[i,0][7]
            self.f[i,0][6]=self.f2[i,0][8]

            self.f[i,self.NY-1][4]=self.f2[i,self.NY-1][2]
            self.f[i,self.NY-1][7]=self.f2[i,self.NY-1][5]
            self.f[i,self.NY-1][8]=self.f2[i,self.NY-1][6]


        #Periodic
        for j in range(self.NY):
            self.f[0,j][1]=self.f[self.NX-1,j][1]
            self.f[0,j][5]=self.f[self.NX-1,j][5]
            self.f[0,j][8]=self.f[self.NX-1,j][8]

            self.f[self.NX-1,j][3]=self.f[0,j][3]
            self.f[self.NX-1,j][6]=self.f[0,j][6]
            self.f[self.NX-1,j][7]=self.f[0,j][7]



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

        self.computePressure()

        self.computeSCForces()

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
        # density=self.rho.to_numpy() 
        pressure=self.pressure.to_numpy()
        return pressure

    def pos_density(self):
        rho=self.rho.to_numpy()-1
        return rho

    def writeVTK(self,fname):
        rho=self.rho.to_numpy().T.flatten()  
        vel=self.vel.to_numpy() 
        velx=vel[:,:,0].T.flatten()  
        vely=vel[:,:,1].T.flatten()  

        bodyforce=self.bodyForce.to_numpy()
        bodyforcex=bodyforce[:,:,0].T.flatten()
        bodyforcey=bodyforce[:,:,1].T.flatten()

        scforce=self.SCforce.to_numpy()
        scfx=scforce[:,:,0].T.flatten()
        scfy=scforce[:,:,1].T.flatten()

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

            np.savetxt(fout, (rho - 1.0) * self.pressure_conversion/3.0, fmt='%.8f') 


            fout.write("VECTORS velocity double\n")  
            velocity_data = np.column_stack((velx * self.velcity_conversion, vely * self.velcity_conversion, np.zeros_like(velx)))  
            np.savetxt(fout, velocity_data, fmt='%.8f') 
  
            fout.write("VECTORS f double\n")  
            bodyforce = np.column_stack((bodyforcex, bodyforcey, np.zeros_like(bodyforcex)))  
            np.savetxt(fout, bodyforce, fmt='%.8f') 

            fout.write("VECTORS scforce double\n")  
            bodyforce = np.column_stack((scfx, scfy, np.zeros_like(scfx)))  
            np.savetxt(fout, bodyforce, fmt='%.8f') 

        print(filename)

