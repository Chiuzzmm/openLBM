import taichi as ti
import taichi.math as tm
import numpy as np
import math


@ti.data_oriented
class Sphere:
    def __init__(self,particles_max_Number,particles_max_NUMIB):
        self.particles_max_Number=particles_max_Number
        self.particles_num=0
        self.particles_id=ti.field(int,shape=(particles_max_Number))
        self.particles_pos=ti.Vector.field(2,float,shape=(particles_max_Number))
        self.particles_vel=ti.Vector.field(2,float,shape=(particles_max_Number))
        self.particles_force=ti.Vector.field(2,float,shape=(particles_max_Number))
        self.particles_torque=ti.field(float,shape=(particles_max_Number))
        self.particles_radius=ti.field(float,shape=(particles_max_Number))
        self.particles_angle=ti.field(float,shape=(particles_max_Number))

        self.particles_max_NUMIB=particles_max_NUMIB
        self.NUMIB_cof=1
        self.particles_IB_nodes_NUMIB=ti.field(int,shape=(particles_max_Number))
        self.particles_IB_nodes_pos=ti.Vector.field(2,float,shape=(particles_max_Number,particles_max_NUMIB))
        self.particles_IB_nodes_vel=ti.Vector.field(2,float,shape=(particles_max_Number,particles_max_NUMIB))
        self.particles_IB_nodes_vel_desired=ti.Vector.field(2,float,shape=(particles_max_Number,particles_max_NUMIB))
        self.particles_IB_nodes_force=ti.Vector.field(2,float,shape=(particles_max_Number,particles_max_NUMIB))
        self.particles_IB_nodes_force_temp=ti.Vector.field(2,float,shape=(particles_max_Number,particles_max_NUMIB))

        self.particles_IB_nodes_mask_insiderDomain=ti.field(int,shape=(particles_max_Number,particles_max_NUMIB))
        self.particles_IB_nodes_mask_insiderGroup=ti.Vector.field(2,int,shape=(particles_max_Number*particles_max_NUMIB))
        self.particles_IB_nodes_mask_insiderCount=ti.field(int,shape=())


    def init_Balls(self,particles_num,id_np,pos_np,radius_np,angle_np):
        self.particles_num=particles_num
        self.particles_id.from_numpy(id_np)
        self.particles_pos.from_numpy(pos_np)
        self.particles_radius.from_numpy(radius_np)
        self.particles_angle.from_numpy(angle_np)
        print("init Balls")

    @ti.kernel
    def init_IB_nodes(self):
        for m in range(self.particles_num):
            self.particles_IB_nodes_NUMIB[m]= int(round(tm.pi * 2 * self.particles_radius[m] / self.NUMIB_cof))
            for n in range(self.particles_IB_nodes_NUMIB[m]):
                angle = 2.0 * math.pi * n / self.particles_IB_nodes_NUMIB[m]
                self.particles_IB_nodes_pos[m,n].x=self.particles_pos[m].x+self.particles_radius[m]*tm.cos(angle)
                self.particles_IB_nodes_pos[m,n].y=self.particles_pos[m].y+self.particles_radius[m]*tm.sin(angle)
        print("init IB nodes")


    @ti.func
    def CheckPos(self,pos,diameter):
        posflag=2 #cross
        if pos.x-diameter>0 and pos.x+diameter<self.NX-1 and pos.y-diameter>0 and pos.y+diameter<=self.NY-1:
            posflag=1 #inside
        elif pos.x+diameter<0 or pos.x-diameter>self.NX-1 or pos.y+diameter<0 or pos.y-diameter>self.NY-1:
            posflag=0 #outside
        return posflag
    
    @ti.func
    def InDomain(self,pos):
        flag=0
        if pos.x>0 and pos.x<self.NX-1 and pos.y>0 and pos.y<self.NY-1:
            flag=1
        return flag

    @ti.func
    def computer_IB_nodes_pos(self,m,n):
        pos=ti.Vector([0.0,0.0])
        angle = 2.0 * math.pi * n / self.particles_IB_nodes_NUMIB[m]
        xref=self.particles_radius[m]*tm.cos(angle) 
        yref=self.particles_radius[m]*tm.sin(angle)

        pos[0]=self.particles_pos[m].x+xref*tm.cos(self.particles_angle[m])-yref*tm.sin(self.particles_angle[m])
        pos[1]=self.particles_pos[m].y+xref*tm.sin(self.particles_angle[m])+yref*tm.cos(self.particles_angle[m])
        return pos
    

@ti.data_oriented
class IBDEMCouplerEngine():
    @ti.kernel
    def SphereIBUpdate(sphere_field:ti.template()):

        sphere_field.particles_IB_nodes_mask_insiderGroup.fill([0,0])
        sphere_field.particles_IB_nodes_mask_insiderCount[None]=0


        for m in ti.static(range(sphere_field.particles_num)):
            posflag=sphere_field.CheckPos(sphere_field.particles_pos[m],2*sphere_field.particles_radius[m])
            if posflag==1:#inside
                for n in ti.static(range(sphere_field.particles_IB_nodes_NUMIB[m])):
                    sphere_field.particles_IB_nodes_pos[m,n]=sphere_field.computer_IB_nodes_pos(m,n)
                    sphere_field.particles_IB_nodes_vel_desired[m,n]=sphere_field.particles_vel[m]
                    idx=ti.atomic_add(sphere_field.particles_IB_nodes_mask_insiderCount[None],1)
                    sphere_field.particles_IB_nodes_mask_insiderGroup[idx]=ti.Vector([m,n])

            elif posflag==2:#cross
                for n in ti.static(range(sphere_field.particles_IB_nodes_NUMIB[m])):
                    sphere_field.particles_IB_nodes_pos[m,n]=sphere_field.computer_IB_nodes_pos(m,n)
                    sphere_field.particles_IB_nodes_vel_desired[m,n]=sphere_field.particles_vel[m]
                    if sphere_field.InDomain(sphere_field.particles_IB_nodes_pos[m,n]):
                        idx=ti.atomic_add(sphere_field.particles_IB_nodes_mask_insiderCount[None],1)
                        sphere_field.particles_IB_nodes_mask_insiderGroup[idx]=ti.Vector([m,n])
                
    def SphereUpdata(self,sphere_field:ti.template(),pos_np,vel_np,angle_np):
        sphere_field.particles_pos.from_numpy(pos_np)
        sphere_field.particles_vel.from_numpy(vel_np)
        sphere_field.particles_angle.from_numpy(angle_np)



@ti.data_oriented
class IBEngine():
    def __init__(self,MDFIteration):
        self.MDFIteration=MDFIteration

    @ti.kernel#IB solve
    def ComputeParticleForce(self,sphere_field:ti.template()):
        particles_IB_nodes_mask_insiderCount=sphere_field.particles_IB_nodes_mask_insiderCount[None]
        for idx in range(particles_IB_nodes_mask_insiderCount):
            m,n=sphere_field.particles_IB_nodes_mask_insiderGroup[idx]
            #reset force
            sphere_field.particles_torque[m]=0.0
            sphere_field.particles_force[m]=ti.Vector([0.0,0.0])

            fx,fy=sphere_field.particles_IB_nodes_force[m,n]
            rx,ry=sphere_field.particles_IB_nodes_pos[m,n]-sphere_field.particles_pos[m]

            sphere_field.particles_torque[m]+=fx*ry-fy*rx
            sphere_field.particles_force[m].x+=fx
            sphere_field.particles_force[m].y+=fy

    @ti.func
    def BottomAndTop(self,pos):
        posArray=ti.Vector([0.0,0.0,0.0,0.0])
        posArray[0]=ti.cast(tm.floor(pos.x),int)
        posArray[1]=ti.cast(tm.floor(pos.y),int)
        posArray[2]=ti.cast(tm.ceil(pos.x),int)
        posArray[3]=ti.cast(tm.ceil(pos.y),int)
        return posArray
    
    @ti.func
    def weight(self,pos,iX,iY):
        return (1.0-ti.abs(pos.x-iX))*(1.0-ti.abs(pos.y-iY))
    

    @ti.kernel#IB solve
    def InterpolateParticleVelocities(self,sphere_field:ti.template(),lb_filed:ti.template()):
        particles_IB_nodes_mask_insiderCount=sphere_field.particles_IB_nodes_mask_insiderCount[None]
        for idx in range(particles_IB_nodes_mask_insiderCount):
            m,n=sphere_field.particles_IB_nodes_mask_insiderGroup[idx]
            #reset vel
            sphere_field.particles_IB_nodes_vel[m,n]=ti.Vector([0.0,0.0])

            # Identify the lowest fluid lattice node in interpolation range (see spreading).
            xbottom,ybottom,xtop,ytop=self.BottomAndTop(sphere_field.particles_IB_nodes_pos[m,n])

            for iX, iY in ti.ndrange((xbottom, xtop + 1), (ybottom, ytop + 1)): 
                # Compute distance between object node and fluid lattice node.
			    # Compute interpolation weights for x- and y-direction based on the distance.
                weight=self.weight(sphere_field.particles_IB_nodes_pos[m,n],iX,iY)

                # Compute node velocities.
                sphere_field.particles_IB_nodes_vel[m,n]+=lb_filed.vel[iX,iY]*weight

            # Compute the Lagrangian correction force
            ForceCorrection=2*(sphere_field.particles_IB_nodes_vel_desired[m,n]-sphere_field.particles_IB_nodes_vel[m,n])
            sphere_field.particles_IB_nodes_force_temp[m,n]=ForceCorrection
            sphere_field.particles_IB_nodes_force[m,n]+=ForceCorrection

    @ti.kernel#IB solve
    def SpreadParticleForces(self,sphere_field:ti.template(),lb_filed:ti.template()):
        #spread forces
        particles_IB_nodes_mask_insiderCount=sphere_field.particles_IB_nodes_mask_insiderCount[None]
        for idx in range(particles_IB_nodes_mask_insiderCount):
            m,n=sphere_field.particles_IB_nodes_mask_insiderGroup[idx]

            xbottom,ybottom,xtop,ytop=self.BottomAndTop(sphere_field.particles_IB_nodes_pos[m,n])

            for iX, iY in ti.ndrange((xbottom, xtop + 1), (ybottom, ytop + 1)): 
                # Compute interpolation weights for x- and y-direction based on the distance.
                weight=self.weight(sphere_field.particles_IB_nodes_pos[m,n],iX,iY)
                #Compute lattice force.
                f_temp=ti.Vector([0.0,0.0])
                f_temp+=sphere_field.particles_IB_nodes_force_temp[m,n]*weight
                #Correct previous Eulerian velocity
                r2=lb_filed.rho[iX,iY]*2
                lb_filed.vel[iX,iY]+=f_temp/r2
                lb_filed.bodyForce[iX,iY]+=f_temp

    def MultiDirectForcingLoop(self,sphere_field:ti.template(),lb_filed:ti.template()):
        sphere_field.particles_IB_nodes_force.fill([0,0])
        for m in range(self.MDFIteration):
            self.InterpolateParticleVelocities(sphere_field=sphere_field,lb_filed=lb_filed)
            self.SpreadParticleForces(sphere_field=sphere_field,lb_filed=lb_filed)
