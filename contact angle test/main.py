import LBIB_taichi
import numpy as np
import taichi as ti
import math
from matplotlib import cm
import time
import matplotlib 

ti.init(arch=ti.gpu)


NX_LB =int(200)
NY_LB =int(200)
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
    for j in range(20):

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
