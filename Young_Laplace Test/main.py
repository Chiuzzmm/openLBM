import LBIB_taichi
import numpy as np
import taichi as ti
import math
from matplotlib import cm
import time
import matplotlib 

ti.init(arch=ti.gpu)

#mesh 
scale=1e-3
domainx=64*scale
domainy=64*scale
NX_LB =int(64)
NY_LB =int(64)
Cl=domainx/NX_LB


# flow boundary condition
Umax=1e-2
pressure_lnlet=10

# flow info
#nu=1e-6# Kinematic viscosity
rho_flow=1000#it.fish.get('rho_ball')#1000 # density of fluid

cs=0.578 # sound speed 
Ma=Umax/cs*1.5 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0
U_LB=max(Ux_LB,Uy_LB)
Cu=Umax/U_LB #vel conversion (main)
Ct=Cl/Cu #time conversion

Cnu=Cl**2/Ct  #Kinematic viscosity conversion

nu=0.5*Cnu/3.0
tau_sym_LB=nu*3*Ct/Cl**2+0.5
omega=1/tau_sym_LB

C_rho=rho_flow/1 # density conversion (main)
C_force=Cl**3*C_rho/Ct**2 # force conversion
C_torque=C_force*Cl
C_pressure=C_rho*Cu**2

rho_lnlet=1+pressure_lnlet*3/C_pressure

#==============================================
name="test"
test=LBIB_taichi.LBIBForm(name=name,NX=NX_LB,NY=NY_LB)


#LBM init
#==============================================LBM boundary
test.Boundary_identify()
test.writing_boundary() #cheack boundary


test.init_hydro()
test.init_LBM()

#==============================================solve & show

start_time = time.time()
gui = ti.GUI(name, (NX_LB,2*NY_LB)) 

index=0
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
# for i in range (10):
    for j in range(2):

        # LBM SOLVE
        test.LBIB_solve()

    pressure = cm.coolwarm(test.post_pressure())
    vel_img = cm.plasma(test.post_vel()/0.1)
    img = np.concatenate((pressure, vel_img), axis=1)

    gui.set_image(img)
    gui.show()

    
    # time.sleep(0.2)
    # filename="test"+'%d' % i
    # test.writeVTK(filename)
    # end_time = time.time()
    # elapsed_time = (end_time - start_time)
    # print({elapsed_time})
