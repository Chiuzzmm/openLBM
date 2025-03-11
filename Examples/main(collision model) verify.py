
import numpy as np
import taichi as ti
import openLBM
import numpy as np
from matplotlib import cm
import time

#couette flow

ti.init(arch=ti.gpu)

#mesh 
scale=1
domainx=40*scale
domainy=20*scale
NX_LB =int(40)
NY_LB =int(20)
Cl=domainx/NX_LB


# flow boundary condition
Umax=0.1


# flow info
shear_viscosity=(2*0.9-1)/6
bulk_viscosity=shear_viscosity*2.5
rho_flow=1#it.fish.get('rho_ball')#1000 # density of fluid

#conversion coefficient
Ux_LB=0.1# vel of LB
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
omega_v=1/(shear_viscosity_LB*3.0+0.5)
omega_e=1/(bulk_viscosity_LB*3.0+0.5)
omega_q=1.0
omega_epsilon=1.0


C_rho=rho_flow/1 # density conversion (main)
C_force=Cl**3*C_rho/Ct**2 # force conversion
C_torque=C_force*Cl
C_pressure=C_rho*Cu**2


#==============================================
name="test"
lb_field=openLBM.LBField(name,NX_LB,NY_LB)
lb_field.BoundaryInfo(InletMode=1,vel_Inlet=ti.Vector([Ux_LB,Uy_LB]))

#post pars
lb_field.conversion_coefficient(Cu_py=Cu,C_pressure_py=C_pressure)
#==============================================
boundary_engine=openLBM.BoundaryEngine()

boundary_engine.Boundary_identify(lb_field)
boundary_engine.writing_boundary(lb_field)


#==============================================
macroscopic_engine=openLBM.MacroscopicEngine()
#==============================================
# collision_engine=openLBM.BGKCollision()
# collision_engine.Relaxation_pars(omega=omega)#BGK

# collision_engine=openLBM.TRTCollision()
# collision_engine.Relaxation_pars(omega_sym=omega_sym_LB,omega_antisym=omega_antisym_LB)#TRT

collision_engine=openLBM.MRTCollision()
collision_engine.Relaxation_pars(omega_v=omega_v,omega_e=omega_e,omega_q=omega_q,omega_epsilon=omega_epsilon)#MRT

#==============================================

lb_field.init_hydro()
lb_field.init_LBM(collision_engine)


#==============================================
stream_engine=openLBM.StreamEngine()

#==============================================solve & show

start_time = time.time()
gui = ti.GUI(name, (NX_LB,2*NY_LB)) 

while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    for i in range (10):
        for j in range(2000):

            # LBM SOLVE
            macroscopic_engine.computeDensity(lb_field)
            macroscopic_engine.computerForceDensity(lb_field)
            macroscopic_engine.computeVelocity(lb_field)
            
            collision_engine.Collision(lb_field)

            stream_engine.StreamPeriodic(lb_field)

            boundary_engine.BB(lb_field)

        
        pressure = cm.coolwarm(lb_field.post_pressure())
        vel_img = cm.plasma(lb_field.post_vel()/0.05)
        img = np.concatenate((pressure, vel_img), axis=1)

        gui.set_image(img)
        gui.show()

        filename="test"+'%d' % i
        lb_field.writeVTK(filename)
        end_time = time.time()
        elapsed_time = (end_time - start_time)
        print({elapsed_time})

        