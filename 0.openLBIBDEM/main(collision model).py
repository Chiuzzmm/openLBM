
import numpy as np
import taichi as ti
import openLBM
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
Umax=1e-2
pressure_lnlet=10

# flow info
shear_viscosity=1e-6
bulk_viscosity=1.28e-6
rho_flow=1000#it.fish.get('rho_ball')#1000 # density of fluid

cs=0.578 # sound speed 
Ma=Umax/cs*1.5 #The larger the Ma, the larger the time step


#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
U_LB=max(Ux_LB,Uy_LB)
Cu=Umax/U_LB #vel conversion (main)
Ct=Cl/Cu #time conversion

Cnu=Cl**2/Ct  #Kinematic viscosity conversion

#BGK
# tau_sym_LB=shear_viscosity*3*Ct/Cl**2+0.5
# omega=1/tau_sym_LB

#TRT
# Magic=1/4
# tau_sym_LB=shear_viscosity*3*Ct/Cl**2+0.5
# tau_antisym_LB=Magic/(tau_sym_LB-0.5)+0.5
# omega_sym_LB=1/tau_sym_LB
# omega_antisym_LB=1/tau_antisym_LB


#MRT
shear_viscosity_LB=shear_viscosity/Cnu
bulk_viscosity_LB=bulk_viscosity/Cnu
omega_v=1/(shear_viscosity_LB*3+0.5)
omega_e=1/(bulk_viscosity_LB*3+0.5)
omega_q=1.0
omega_epsilon=1.0


C_rho=rho_flow/1 # density conversion (main)
C_force=Cl**3*C_rho/Ct**2 # force conversion
C_torque=C_force*Cl
C_pressure=C_rho*Cu**2

rho_lnlet=1+pressure_lnlet*3/C_pressure

#==============================================
name="test"
lb_field=openLBM.LBField(name,NX_LB,NY_LB)
lb_field.BoundaryInfo(InletMode=1,vel_Inlet=ti.Vector([Ux_LB,Uy_LB]))

#post pars
lb_field.conversion_coefficient(Cu_py=Cu,C_pressure_py=C_pressure)
#==============================================
boundary_engine=openLBM.BoundaryEngine()
boundary_engine.Mask_rectangle_identify(lb_field,p6[0]/Cl-0.5,p2[0]/Cl-0.5,p6[1]/Cl-0.5,p2[1]/Cl-0.5)
boundary_engine.Mask_rectangle_identify(lb_field,p4[0]/Cl-0.5,p8[0]/Cl-0.5,p4[1]/Cl-0.5,p8[1]/Cl-0.5)
boundary_engine.Boundary_identify(lb_field)
boundary_engine.writing_boundary(lb_field)

#==============================================
global_engine=openLBM.GlobalEngine()
global_engine.init_hydro(lb_field)
global_engine.init_LBM(lb_field)
#==============================================
macroscopic_engine=openLBM.MacroscopicEngine()
#==============================================
collision_engine=openLBM.CollisionEngine()
# collision_engine.Relaxation_pars(omega=omega)#BGK
# collision_engine.Relaxation_pars(omega_sym=omega_sym_LB,omega_antisym=omega_antisym_LB)#TRT
collision_engine.Relaxation_pars(omega_e=omega_e,omega_v=omega_v)

#==============================================
stream_engine=openLBM.StreamEngine()

#==============================================solve & show

start_time = time.time()
gui = ti.GUI(name, (NX_LB,2*NY_LB)) 

while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
# for i in range (10):
    for j in range(20):

        # LBM SOLVE
        macroscopic_engine.computeDensity(lb_field)
        macroscopic_engine.computerForceDensity(lb_field)
        macroscopic_engine.computeVelocity(lb_field)
        
        # collision_engine.BGKCollide(lb_field)
        # collision_engine.TRTCollide(lb_field)
        collision_engine.MRTCollide(lb_field)

        stream_engine.StreamInside(lb_field)
        
        boundary_engine.BounceBackInside(lb_field)
        boundary_engine.BounceBackInlet(lb_field)
        boundary_engine.BounceBackOutlet(lb_field)

        
    pressure = cm.coolwarm(lb_field.post_pressure())
    vel_img = cm.plasma(lb_field.post_vel()/0.15)
    img = np.concatenate((pressure, vel_img), axis=1)

    gui.set_image(img)
    gui.show()

    # filename="test"+'%d' % i
    # global_engine.writeVTK(filename,lb_field)
    # end_time = time.time()
    # elapsed_time = (end_time - start_time)
    # print({elapsed_time})