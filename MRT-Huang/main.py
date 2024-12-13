import LBIB_taichi
import numpy as np
import taichi as ti
import math
from matplotlib import cm
import time
import matplotlib 

ti.init(arch=ti.gpu)

#mesh 
NX_LB =int(64)
NY_LB =int(64)


omega_v=1.0
omega_e=omega_v
omega_epsilon=1.0
omega_q=1.0


#==============================================
name="test"
test=LBIB_taichi.LBIBForm(name=name,NX=NX_LB,NY=NY_LB)


#LBM init
test.init_MRT(omega_v=omega_v,omega_e=omega_e,omega_epsilon=omega_epsilon,omega_q=omega_q)
test.init_SC(epslion=2,k1=0.15)
#==============================================LBM boundary
test.Boundary_identify()
test.writing_boundary() #cheack boundary


test.init_hydro()
test.init_LBM()
#=============================================solve & show

start_time = time.time()

gui = ti.GUI(name, (NX_LB,2*NY_LB)) 
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
# for i in range (10):
    for j in range(200):
        # LBM SOLVE
        test.LBIB_solve()

    pressure = cm.coolwarm(test.post_pressure())
    vel_img = cm.plasma(test.post_vel()/0.05)
    img = np.concatenate((pressure, vel_img), axis=1)

    gui.set_image(img)
    gui.show()

    time.sleep(0.1)
    test.info()
    # filename="test"+'%d' % i
    # test.writeVTK(filename)
    # end_time = time.time()
    # elapsed_time = (end_time - start_time)/60
    # print({elapsed_time})
