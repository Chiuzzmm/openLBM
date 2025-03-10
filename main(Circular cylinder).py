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


ti.init(arch=ti.gpu)

#mesh 
scale=1e-3
domainx=20*scale
domainy=8*scale
NX_LB =int(20*20)
NY_LB =int(8*20)
Cl=domainx/NX_LB

Length_XX1=10*scale
Length_YY1=8*scale

p0=[domainx/8*6,Length_YY1/2]


# flow boundary condition
Umax=1e-2
pressure_lnlet=10

# flow info
num_components=1
shear_viscosity=[1e-6]
bulk_viscosity=[2.5e-6]
rho_flow=1000#it.fish.get('rho_ball')#1000 # density of fluid

cs=0.578 # sound speed 
Ma=Umax/cs*2.0 #The larger the Ma, the larger the time step

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
# lb_field.set_gravity(ti.Vector([5e-6,0.0]))

#==============================================
boundary_engine=openLBDEM.BoundaryEngine()
boundary_classifier=openLBDEM.BoundaryClassifier(NX=NX_LB,NY=NY_LB)

boundary_engine.Mask_cricle_identify(lb_field,p0[0]/Cl-0.5,p0[1]/Cl-0.5,Length_YY1/8/Cl-0.5)


def fluid_boundary(i,j):
    return lb_field.mask[i,j]==1

fluid=openLBDEM.BoundarySpec(geometry_fn=fluid_boundary)
fluid_bc=openLBDEM.FluidBoundary(spec=fluid)
fluid_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("fluid",fluid_bc)


stream_bc=openLBDEM.PeriodicAllBoundary(spec=fluid)
stream_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("stream",stream_bc)


def wall_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 :
        for k in ti.static(range(lb_field.NPOP)):
            ix2=i-lb_field.c[k,0]
            iy2=j-lb_field.c[k,1]
            # if ix2<0 or ix2>lb_field.NX-1 or iy2<0 or iy2>lb_field.NY-1 or lb_field.mask[ix2,iy2]==-1:
            if  iy2<0 or iy2>lb_field.NY-1 or lb_field.mask[ix2,iy2]==-1:
            # if lb_field.mask[ix2,iy2]==-1:
                flag=1
    return flag

wall=openLBDEM.BoundarySpec(geometry_fn=wall_boundary)
wall_bc=openLBDEM.BounceBackWall(spec=wall)
wall_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("wall",wall_bc)


def inlet_boundary(i, j):
    flag=0
    if lb_field.mask[i,j]==1 and i==0:
        flag=1
    return flag  

inlet = openLBDEM.BoundarySpec(geometry_fn=inlet_boundary)
velocity_bc=openLBDEM.VelocityBoundary(spec=inlet,velocity_value=ULB,direction=1)
velocity_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("inlet",velocity_bc)

def outlet_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 and i==NX_LB-1:
        flag=1
    return flag 

outlet=openLBDEM.BoundarySpec(geometry_fn=outlet_boundary,direction=1)
pressure_bc=openLBDEM.PressureBoundary(spec=outlet,rho_value=1.0,direction=3)
pressure_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("outlet",pressure_bc)



lb_field.neighbor_classify()
boundary_engine.writing_boundary(lb_field)
#==============================================
macroscopic_engine=openLBDEM.MacroscopicEngine(fluid_bc.group)
#==============================================
# collision_engine=openLBDEM.BGKCollision(num_components,fluid_bc.group)
# collision_engine.unit_conversion(lb_field)

# collision_engine=openLBDEM.TRTCollision(num_components,fluid_bc.group,Magic)
# collision_engine.unit_conversion(lb_field,Magic)

collision_engine=openLBDEM.MRTCollision(num_components,fluid_bc.group)
collision_engine.unit_conversion(lb_field)
# #==============================================
post_processing_engine=openLBDEM.PostProcessingEngine(0)



@ti.kernel 
def init_hydro(vel:ti.types.vector(2, ti.f32),pressure_lnlet:float):
    if pressure_lnlet==0.0:
        for m in range(fluid_bc.group.count[None]):
            ix,iy=fluid_bc.group.group[m]
            lb_field.vel[ix,iy]=vel
            for component in range(lb_field.num_components[None]):
                lb_field.rho[component,ix,iy]=1.0/ lb_field.num_components[None]
    else:
        rho_inlet=1+pressure_lnlet*3/lb_field.C_pressure
        for m in range(fluid_bc.group.count[None]):
            ix,iy=fluid_bc.group.group[m]
            k=(1.0-rho_inlet)/lb_field.NX
            lb_field.vel[ix,iy]=ti.Vector([.0,.0])
            for component in range(lb_field.num_components[None]):
                lb_field.rho[component,ix,iy]=(k*ix+rho_inlet)/ lb_field.num_components[None]
    print("init hydro")
init_hydro(ULB,0.0)

lb_field.init_LBM(collision_engine,fluid_bc.group)


# #==============================================


# ==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)
    
    collision_engine.apply(lb_field)
    
    boundary_engine.apply_boundary_conditions(lb_field)
    
    pass

def post():
    pressure = cm.Blues(post_processing_engine.post_pressure(lb_field))
    vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    img1 = np.concatenate((pressure, vel_img), axis=1)
    return img1

showmode=1 #1=while # 0=iterations
start_time = time.time()
gui = ti.GUI(name, (NX_LB,2*NY_LB)) 

if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for i in range(10):
            for j in range(20):
                lbm_solve()
            img=post()
            gui.set_image(img)
            gui.show()
else:
    video_manager = ti.tools.VideoManager(output_dir="./results", framerate=24, automatic_build=False)
    for i in range(50):
        for j in range(100):
            lbm_solve()
        img=post()
        gui.set_image(img)
        gui.show()
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


# # 在计算结束后显示绘图窗口
# if showmode == 2:
#     plt.ioff()  # 关闭交互模式
#     plt.show()  # 显示绘图窗口并进入事件循环