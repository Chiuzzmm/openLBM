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
shear_viscosity=1e-6 # Kinematic viscosity
bulk_viscosity=1.28e-6
rho=1000

cs=0.578 # sound speed 
Ma=Umax/cs*2.5 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
U_LB=max(Ux_LB,Uy_LB)
Cu=Umax/U_LB #vel conversion (main)
Ct=Cl/Cu #time conversion

Cnu=Cl**2/Ct  #Kinematic viscosity conversion
shear_viscosity_LB=shear_viscosity/Cnu
bulk_viscosity_LB=bulk_viscosity/Cnu

C_rho=rho/1 # density conversion (main)
C_force=Cl**3*C_rho/Ct**2 # force conversion
C_torque=C_force*Cl
C_pressure=C_rho*Cu**2


rho_lnlet=1+pressure_lnlet*3/C_pressure

#MRT control pars
omega_v=1/(shear_viscosity_LB*3+0.5)
omega_e=1/(bulk_viscosity_LB*3+0.5)

omega_q=1.0
omega_epsilon=1.0

# particles info
Number = 1

id_np=arr = np.arange(0,Number, 1) 
pos_np = np.array([[100,120]]).astype(np.float32)
vel_np = np.array([[0,0]]).astype(np.float32)
force_np=np.array([[0,0]]).astype(np.float32)
torque_np= np.array([0]).astype(np.float32)
radius_np = np.array([20]).astype(np.float32)
angle_np=np.array([0]).astype(np.float32)


#==============================================
name="test"
test=LBIB_taichi.LBIBForm(name=name,NX=NX_LB,NY=NY_LB,particles_Number=Number)

test.init_Balls(id_np=id_np,pos_np=pos_np,radius_np=radius_np,angle_np=angle_np)
test.init_IB_nodes()

#LBM init


InletMode=1 #1=vel,2=rho,3=period
test.boundaryInfo(InletMode=InletMode,vx=Ux_LB,vy=Uy_LB,p=rho_lnlet)
test.conversion_coefficient(Cu_py=Cu,C_pressure_py=C_pressure)
test.init_MRT(omega_v=omega_v,omega_e=omega_e,omega_epsilon=omega_epsilon,omega_q=omega_q)

#==============================================LBM boundary
# test.Mask_rectangle_identify(p6[0]/Cl-0.5,p2[0]/Cl-0.5,p6[1]/Cl-0.5,p2[1]/Cl-0.5)
# test.Mask_rectangle_identify(p4[0]/Cl-0.5,p8[0]/Cl-0.5,p4[1]/Cl-0.5,p8[1]/Cl-0.5)

test.Boundary_identify()
test.writing_boundary() #cheack boundary


test.init_hydro()
test.init_LBM()


#=============================================solve & show

start_time = time.time()

gui = ti.GUI(name, (NX_LB,2*NY_LB)) 
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
# for i in range (10):

    start_time = time.time()
    for j in range(50):
        test.BallsInfoUpdata(pos_np=pos_np,vel_np=vel_np,angle_np=angle_np)
        test.BallsIBUpdate()

        # LBM SOLVE
        test.LBIB_solve()

        end_time = time.time()
        elapsed_time = (end_time - start_time)/60
        print({elapsed_time})

    pressure = cm.coolwarm(test.post_pressure())
    vel_img = cm.plasma(test.post_vel()/0.05)
    img = np.concatenate((pressure, vel_img), axis=1)

    gui.set_image(img)
    gui.show()

    # time.sleep(0.2)
    # filename="test"+'%d' % i
    # test.writeVTK(filename)

