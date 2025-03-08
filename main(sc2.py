import sys
import os

# 添加包的路径
sys.path.append(os.path.join(os.path.dirname(__file__), "openLBM"))


import taichi as ti
import openLBDEM
import numpy as np
from matplotlib import cm
import time
import matplotlib.pyplot as plt

print("\n" * 50)
ti.init(arch=ti.gpu)

#mesh 
scale=1e-3
domainx=1*scale
domainy=1*scale
NX_LB =int(20*10)+1
NY_LB =int(20*10)+1
Cl=domainx/(NX_LB-1)


# flow boundary condition
Umax=1e-3

# flow info
num_components=2

shear_viscosity=np.array([1e-6,1.5e-5])
bulk_viscosity=np.array([1e-6,1e-6])
# shear_viscosity=np.array([1e-6,1.5e-5])
# bulk_viscosity=np.array([1e-6,1.5e-5])
rho_l=1#it.fish.get('rho_ball')#1000 # density of fluid
rho_g=1
# rho0=np.sqrt(rho_l*rho_g)
rho0=1

cs=0.578 # sound speed 
Ma=Umax/cs*2.0 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho0/1 # density conversion (main)



#TRT
Magic=np.array([1/4,1/4])



gA=-0.0
gB=-0.0
gAB=6.0
g=np.array([[gA,gAB],[gAB,gB]])

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
sc_engine=openLBDEM.ShanChenForceC2(lb_field,g)
#==============================================
# collision_engine=openLBDEM.BGKCollision(num_components)
# collision_engine.unit_conversion(lb_field)


# collision_engine=openLBDEM.TRTCollision(num_components)
# collision_engine.unit_conversion(lb_field,Magic)

collision_engine=openLBDEM.MRTCollision(num_components)
collision_engine.unit_conversion(lb_field)
#==============================================
post_processing_engine=openLBDEM.PostProcessingEngine(1)


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
    
def post():
    pressure = cm.Blues(post_processing_engine.post_pressure(lb_field))
    vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    img1 = np.concatenate((pressure, vel_img), axis=1)
    img2=cm.magma(post_processing_engine.post_MC_pressure(lb_field))
    img=np.concatenate((img1,img2),axis=0)
    return img


def history():
    x1 = np.arange(NX_LB)
    p=lb_field.total_pressure.to_numpy()[:,lb_field.NY//2]
    

    # vel = lb_field.vel.to_numpy()
    # vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
    # v=vel_mag[:,lb_field.NY//2]

    post_processing_engine.update_plot([(x1,p)])




post_processing_engine.show()

showmode=1 #1=while # 0=iterations
start_time = time.time()
gui = ti.GUI(name, (2*NX_LB,2*NY_LB)) 


if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for i in range(10):
            for j in range(100):
                lbm_solve()
            img=post()
            gui.set_image(img)
            gui.show()
        history()
        print(f'\rDelta p= {lb_field.total_pressure[100,100]-lb_field.total_pressure[0,100]}', end='')

else:
    video_manager = ti.tools.VideoManager(output_dir="./results", framerate=24, automatic_build=False)
    for i in range(50):
        for j in range(100):
            lbm_solve()
        img=post()
        gui.set_image(img)
        gui.show()
        history()
        print(f'\rDelta p= {lb_field.total_pressure[100,100]-lb_field.total_pressure[0,100]}', end='')

        filename="test"+'%d' % i
        # savefilename = f'2C_unMix_{i:05d}.png'   # create filename with suffix png
        # gui.show(savefilename)
        # post_processing_engine.writeVTK(filename,lb_field)
        # end_time = time.time()
        # elapsed_time = (end_time - start_time)
        # print({elapsed_time})


        video_manager.write_frame(img)
    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=False)
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')


# 在计算结束后显示绘图窗口
if showmode == 2:
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示绘图窗口并进入事件循环