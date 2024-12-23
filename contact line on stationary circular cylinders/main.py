import LBIB_taichi
import numpy as np
import taichi as ti
import math
from matplotlib import cm
import time
import matplotlib 

ti.init(arch=ti.gpu)

#mesh 

NX_LB =int(20*10)
NY_LB =int(20*10)


#MRT control pars
omega_v=0.6
omega_e=omega_v
omega_epsilon=1.0
omega_q=1.0


# particles info
Number = 1
id_np=arr = np.arange(0,Number, 1) 
pos_np = np.array([[200.0,40.0]]).astype(np.float32)
vel_np = np.array([[0.0,0.0]]).astype(np.float32)
force_np=np.array([[0.0,0.0]]).astype(np.float32)
torque_np= np.array([0.0]).astype(np.float32)
radius_np = np.array([10.0]).astype(np.float32)
angle_np=np.array([0.0]).astype(np.float32)


#==============================================
name="test"
test=LBIB_taichi.LBIBForm(name=name,NX=NX_LB,NY=NY_LB)

#LBM init
test.init_MRT(omega_v=omega_v,omega_e=omega_e,omega_epsilon=omega_epsilon,omega_q=omega_q)
test.init_SC(epslion=1.5,k1=0.15)
#==============================================LBM boundary
test.Mask_cricle_identify(x=100,y=100,r=20)

test.Boundary_identify()
test.writing_boundary() #cheack boundary


test.init_hydro()
test.init_LBM()


#=============================================solve & show

start_time = time.time()

gui = ti.GUI(name, (NX_LB,3*NY_LB)) 
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
# for i in range (10):
    start_time = time.time()
    for j in range(5):
        # LBM SOLVE
        test.LBIB_solve()

    pressure = cm.coolwarm(test.post_pressure())
    vel_img = cm.plasma(test.post_vel()/0.05)
    rho_img=cm.coolwarm(test.pos_density())
    
    img = np.concatenate((pressure, vel_img,rho_img), axis=1)

    gui.set_image(img)
    gui.show()

    # time.sleep(0.02)
    # filename="test"+'%d' % i
    # test.writeVTK(filename)

