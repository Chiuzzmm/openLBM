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
domainx=40*scale
domainy=8*scale
NX_LB =int(40*30)
NY_LB =int(8*30)
Cl=domainx/NX_LB

Length_XX1=25*scale
Length_XX2=35*scale
Length_XX3=40*scale
  
Length_YY1=8*scale
Length_YY2=3*scale
Length_YY3=5*scale

p2=[Length_XX2,Length_YY1]
p6=[Length_XX1,Length_YY3]
p4=[Length_XX1,0]
p8=[Length_XX2,Length_YY2]


# flow boundary condition
InletMode=1
Umax=1e-2
pressure_lnlet=10

# flow info
num_components=1
shear_viscosity=[1e-6]
bulk_viscosity=[2.5e-6]
rho_flow=1000#it.fish.get('rho_ball')#1000 # density of fluid

cs=0.578 # sound speed 
Ma=Umax/cs*1.5 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho_flow/1 # density conversion (main)

#TRT
Magic=[1/4]


#==============================================
name="test"
lb_field=openLBDEM.LBField(name,NX_LB,NY_LB,num_components)

#unit 
lb_field.init_conversion(Cl=Cl,Ct=Ct,Crho=C_rho,shear_viscosity=shear_viscosity,bulk_viscosity=bulk_viscosity)


#==============================================
boundary_engine=openLBDEM.BoundaryEngine()
boundary_engine.Mask_rectangle_identify(lb_field,p6[0]/Cl-0.5,p2[0]/Cl-0.5,p6[1]/Cl-0.5,p2[1]/Cl-0.5)
boundary_engine.Mask_rectangle_identify(lb_field,p4[0]/Cl-0.5,p8[0]/Cl-0.5,p4[1]/Cl-0.5,p8[1]/Cl-0.5)
boundary_engine.boundary_identify(lb_field)
boundary_engine.writing_boundary(lb_field)
boundary_engine.add_boundary_condition(openLBDEM.BounceBackWall(lb_field.wall_boundary))
boundary_engine.add_boundary_condition(openLBDEM.VelocityInlet(lb_field.inlet_boundary,ULB))
boundary_engine.add_boundary_condition(openLBDEM.PressureOutlet(lb_field.outlet_boundary,1.0))

#==============================================
macroscopic_engine=openLBDEM.MacroscopicEngine()
#==============================================
# collision_engine=openLBDEM.BGKCollision(num_components)
# collision_engine.unit_conversion(lb_field)

# collision_engine=openLBDEM.TRTCollision(num_components)
# collision_engine.unit_conversion(lb_field,Magic)

collision_engine=openLBDEM.MRTCollision(num_components)
collision_engine.unit_conversion(lb_field)


#==============================================
lb_field.init_hydro(ULB,0.0)
lb_field.init_LBM(collision_engine)


#==============================================


# ==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)
    
    collision_engine.apply(lb_field)

    collision_engine.stream_inside(lb_field)
    
    boundary_engine.apply_boundary_conditions(lb_field)
    
    pass

start_time = time.time()
gui = ti.GUI(name, (NX_LB,2*NY_LB)) 

# while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
for i in range (5):
    for j in range(20):
        lbm_solve()



    pressure = cm.Blues(macroscopic_engine.post_pressure(lb_field)/0.05)
    vel_img = cm.plasma(macroscopic_engine.post_vel(lb_field))
    img1 = np.concatenate((pressure, vel_img), axis=1)

    gui.set_image(img1)
    gui.show()

    filename="test"+'%d' % i
    macroscopic_engine.writeVTK(filename,lb_field)
    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print({elapsed_time})