import sys
import os

# 添加包的路径
sys.path.append(os.path.join(os.path.dirname(__file__), "openLBM"))


import taichi as ti
import openLBDEM
import numpy as np
from matplotlib import cm
import time


ti.init(arch=ti.gpu)

#mesh 
scale=1e-3
domainx=1*scale
domainy=1*scale
NX_LB =int(20*10)
NY_LB =int(20*10)
Cl=domainx/NX_LB


# flow boundary condition
InletMode=1
Umax=1e-3

# flow info
num_components=1
shear_viscosity=np.array([1e-6])
bulk_viscosity=np.array([1e-6])
rho0=1

cs=0.578 # sound speed 
Ma=Umax/cs*1.1 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho0/1 # density conversion (main)



#TRT
Magic=np.array([1/4])

#MRT



gA=-5.0
#==============================================
name="test"
lb_field=openLBDEM.LBField(name,NX_LB,NY_LB,num_components)
#unit 
lb_field.init_conversion(Cl=Cl,Ct=Ct,Crho=C_rho,shear_viscosity=shear_viscosity,bulk_viscosity=bulk_viscosity)
# lb_field.set_gravity(ti.Vector([0.0,-1e-6]))


#==============================================
boundary_engine=openLBDEM.BoundaryEngine()
# boundary_engine.Mask_cricle_identify(lb_field,100,100,20)
boundary_engine.boundary_identify(lb_field)
boundary_engine.writing_boundary(lb_field)

#==============================================
macroscopic_engine=openLBDEM.MacroscopicEngine()
#==============================================
sc_engine=openLBDEM.ShanChenForceC1(lb_field,gA)

#==============================================
# collision_engine=openLBDEM.BGKCollision(num_components)
# collision_engine.unit_conversion(lb_field)


# collision_engine=openLBDEM.TRTCollision(num_components,Magic)
# collision_engine.unit_conversion(lb_field,Magic)

collision_engine=openLBDEM.MRTCollision(num_components)
collision_engine.unit_conversion(lb_field)

# k1=np.array([1.0,1.0])
# epslion=np.array([1.0,1.0])
# collision_engine=openLBDEM.HuangMRTCollision(num_components,k1,epslion)


#==============================================
# lb_field.init_hydro(ULB,0.0)
sc_engine.init_hydro(lb_field)
lb_field.init_LBM(collision_engine)


#==============================================


# ==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field,sc_engine)
    sc_engine.apply(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)
    
    collision_engine.apply(lb_field)

    collision_engine.stream_periodic(lb_field)
    
    # boundary_engine.apply_boundary_conditions(lb_field)
    

start_time = time.time()
gui = ti.GUI(name, (1*NX_LB,2*NY_LB)) 

while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    for i in range (1):
        for j in range(10):
            lbm_solve()

        pressure = cm.Blues(macroscopic_engine.post_pressure(lb_field))
        vel_img = cm.plasma(macroscopic_engine.post_vel(lb_field))
        img1 = np.concatenate((pressure, vel_img), axis=1)

    
        # img2=cm.magma(macroscopic_engine.post_MC_pressure(lb_field))
        # img=np.concatenate((img1,img2),axis=0)
        gui.set_image(img1)
        gui.show()


        # filename="test"+'%d' % i
        # macroscopic_engine.writeVTK(filename,lb_field)
        # end_time = time.time()
        # elapsed_time = (end_time - start_time)
        # print({elapsed_time})