import taichi as ti
import numpy as np
import taichi.math as tm


class Collision:
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
class BGKCollision(Collision):
    def __init__(self):
        super().__init__()
        self.omega=1.0
        self.tau_sym_LB=1.0

    def relaxation_pars(self,omega):
        self.omega=omega

    def unit_conversion(self,lb_field:ti.template()):
        self.tau_sym_LB=lb_field.shear_viscosity_LB*3+0.5
        self.omega=1/self.tau_sym_LB

    @ti.kernel#LBM solve
    def apply(self,lb_field:ti.template()):
        for idx in range(lb_field.fluid_boundary.count[None]):
            i,j = lb_field.fluid_boundary.group[idx]

            feqeq=self.f_eq(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j])
            force_ij=self.f_force(lb_field.c,lb_field.weights,lb_field.body_force[i,j],lb_field.vel[i,j])

            collision_operator =-self.omega*(lb_field.f[i,j]-feqeq)
            force_term=(1.0-0.5*self.omega)*force_ij

            lb_field.f2[i,j]=lb_field.f[i,j]+collision_operator+force_term

@ti.data_oriented
class TRTCollision(Collision):
    def __init__(self):
        super().__init__()
        self.magic=1/4
        self.tau_sym=1.0
        self.tau_antisym=1.0
        self.omega_sym=2.0
        self.omega_antisym=1.0
    
    def relaxation_pars(self,omega_sym,omega_antisym):
        self.omega_sym=omega_sym
        self.omega_antisym=omega_antisym


    def unit_conversion(self,lb_field:ti.template(),magic):
        self.magic=magic
        self.tau_sym_LB=lb_field.shear_viscosity_LB*3+0.5
        self.tau_antisym_LB=self.magic/(self.tau_sym_LB-0.5)+0.5
        self.omega_sym=1.0/self.tau_sym_LB
        self.omega_antisym=1.0/self.tau_antisym_LB

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
    def apply(self,lb_field:ti.template()):
        for idx in range(lb_field.fluid_boundary.count[None]):
            i,j = lb_field.fluid_boundary.group[idx]

            feqeq=self.f_eq(lb_field.c,lb_field.weights,lb_field.vel[i,j],lb_field.rho[i,j])
            force_ij=self.f_force(lb_field.c,lb_field.weights,lb_field.body_force[i,j],lb_field.vel[i,j])

            #symmetrical and antisymmetrical particle distribution functions
            f_sym=self.f_sym(lb_field.f[i,j])
            feq_sym=self.feq_sym(feqeq)
            force_sym=self.force_sym(force_ij)

            f_antisym=lb_field.f[i,j]-f_sym
            feq_antisym=feqeq-feq_sym
            force_antisym=force_ij-force_sym

            collision_operator =-self.omega_sym * (f_sym-feq_sym)-self.omega_antisym*(f_antisym-feq_antisym)
            force_term=(1.0-0.5*self.omega_sym)*force_sym+(1.0-0.5*self.omega_antisym)*force_antisym
            lb_field.f2[i,j]=lb_field.f[i,j]+collision_operator+force_term

@ti.data_oriented
class MRTCollision(Collision):
    def __init__(self):
        super().__init__()
        self.M=ti.field(float,shape=(9,9))
        self.M_inverse=ti.field(float,shape=(9,9))
        M_np =np.array([
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0],
                [4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
                [0.0, -2.0, 0.0, 2.0, 0.0, 1.0, -1.0,-1.0, 1.0],
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

    def relaxation_pars(self,omega_e,omega_v,omega_q=1,omega_epsilon=1):
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

    def unit_conversion(self,lb_field:ti.template(),omega_q=1,omega_epsilon=1):
        self.diag.fill(1.0)
        self.diag[7]=self.diag[8]=1/(lb_field.shear_viscosity_LB*3.0+0.5)
        self.diag[1]=1/(lb_field.bulk_viscosity_LB*3.0+0.5)
        # omega_q=1.0
        # omega_epsilon=1.0


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
    def apply(self,lb_field:ti.template()):
        for idx in range(lb_field.fluid_boundary.count[None]):
            i,j = lb_field.fluid_boundary.group[idx]

            m=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            a=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            b=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            fpop2=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

            #Transform to moment space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    m[ii]+=self.M[ii,jj]*lb_field.f[i,j][jj]

            meq=self.m_eq(lb_field.vel[i,j],lb_field.rho[i,j]) #Compute equilibrium moments
            mf=self.m_force(lb_field.body_force[i,j],lb_field.vel[i,j]) #Guo Forcing

            for ii in ti.static(range(9)):
                a[ii]=self.diag[ii]*(m[ii]-meq[ii])
                b[ii]=(1.0-self.diag[ii]/2.0)*(mf[ii])
            
            m2=m-a+b#Collide

            #Transform to population space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    ti.atomic_add(fpop2[ii],self.M_inverse[ii,jj]*m2[jj])

            lb_field.f2[i,j]=fpop2

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
    def apply(self,lb_field:ti.template()):
        for idx in range(lb_field.fluid_boundary.count[None]):
            i,j = lb_field.fluid_boundary.group[idx]

            m=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            a=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            b=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            c=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            fpop2=ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

            #Transform to moment space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    ti.atomic_add(m[ii],self.M[ii,jj]*lb_field.f[i,j][jj])

            meq=self.m_eq(lb_field.vel[i,j],lb_field.rho[i,j]) #Compute equilibrium moments
            mf=self.m_force(lb_field.body_force[i,j],lb_field.vel[i,j]) #Guo Forcing
            mQ=self.m_Q(lb_field.body_force[i,j],lb_field.rho[i,j]) # huang

            for ii in ti.static(range(9)):
                a[ii]=self.diag[ii]*(m[ii]-meq[ii])
                b[ii]=(1.0-self.diag[ii]/2.0)*(mf[ii])
                c[ii]=self.diag[ii]*mQ[ii]

            m2=m-a+b+c#Collide

            #Transform to population space
            for ii in ti.static(range(9)):
                for jj in ti.static(range(9)):
                    ti.atomic_add(fpop2[ii],self.M_inverse[ii,jj]*m2[jj])

            lb_field.f2[i,j]=fpop2

    @ti.kernel
    def computeSCForces(self,lb_field:ti.template()):
        for m in range(lb_field.fluid_boundary.count[None]):
            i,j = lb_field.fluid_boundary.group[m]

            lb_field.SCforce[i,j].x=0.0
            lb_field.SCforce[i,j].y=0.0

            f_temp_x=0.0
            f_temp_y=0.0
            psinb=0.0
            for k in ti.static(range(1,9)):
                # x2=i+lb_field.c[k,0]
                # y2=j+lb_field.c[k,1]
                x2=(i+lb_field.c[k,0]+lb_field.NX)%lb_field.NX 
                y2=(j+lb_field.c[k,1]+lb_field.NY)%lb_field.NY

                if lb_field.mask[x2,y2]==-1: #solid
                    # if lb_field.rho[i,j]>lb_field.rho_liq/8.0:
                    #     psinb=lb_field.psi(lb_field.rho_solid)
                    # else:
                    psinb=lb_field.psi(lb_field.rho_solid)
                else:
                    psinb=lb_field.psi(lb_field.rho[x2,y2])

                f_temp_x+=lb_field.weights[k]*lb_field.c[k,0]*psinb
                f_temp_y+=lb_field.weights[k]*lb_field.c[k,1]*psinb

            psiloc=lb_field.psi(lb_field.rho[i,j])
            f_temp_x*=(-lb_field.gA*psiloc)
            f_temp_y*=(-lb_field.gA*psiloc)

            ti.atomic_add(lb_field.SCforce[i,j].x,f_temp_x)
            ti.atomic_add(lb_field.SCforce[i,j].y,f_temp_y)