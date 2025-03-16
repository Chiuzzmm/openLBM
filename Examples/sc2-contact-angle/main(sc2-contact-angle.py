import sys
import os

# 添加包的路径
sys.path.append(os.path.join(os.path.dirname(__file__), "openLBM"))

import taichi as ti
import openLBDEM
import numpy as np
from matplotlib import cm
import time
import taichi.math as tm
import math
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
shear_viscosity=np.array([1e-6,1e-6])
bulk_viscosity=np.array([1e-6,1e-6])
rho0=1

cs=0.578 # sound speed 
Ma=Umax/cs*1.5 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho0/1 # density conversion (main)



#TRT
Magic=np.array([1/4,1/4])

#MRT



#==============================================
name="test"
lb_field=openLBDEM.LBField(name,NX_LB,NY_LB,num_components)
#unit 
lb_field.init_conversion(Cl=Cl,Ct=Ct,Crho=C_rho,shear_viscosity=shear_viscosity,bulk_viscosity=bulk_viscosity)
# lb_field.set_gravity(ti.Vector([0.0,-1e-5]))
#==============================================
boundary_engine=openLBDEM.BoundaryEngine()
boundary_classifier=openLBDEM.BoundaryClassifier(NX=NX_LB,NY=NY_LB)
# boundary_engine.Mask_cricle_identify(lb_field,0.5*NX_LB,0.0*NY_LB,0.25*NX_LB)

@ti.func
def fluid_boundary(i,j):
    return lb_field.mask[i,j]==1

fluid=openLBDEM.BoundarySpec(geometry_fn=fluid_boundary)
fluid_bc=openLBDEM.FluidBoundary(spec=fluid)
fluid_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("fluid",fluid_bc)

stream_bc=openLBDEM.PeriodicAllBoundary(spec=fluid)
stream_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("stream",stream_bc)

@ti.func
def wall_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 :
        for k in ti.static(range(lb_field.NPOP)):
            ix2=i-lb_field.c[k,0]
            iy2=j-lb_field.c[k,1]
            if iy2<0 or iy2>lb_field.NY-1 or lb_field.mask[ix2,iy2]==-1:
                flag=1
    return flag

wall=openLBDEM.BoundarySpec(geometry_fn=wall_boundary)
wall_bc=openLBDEM.BounceBackWall(spec=wall)
wall_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("wall",wall_bc)
lb_field.neighbor_classify()
boundary_engine.writing_boundary(lb_field)

#==============================================
macroscopic_engine=openLBDEM.MacroscopicEngine(fluid_bc.group)


#==============================================
# collision_engine=openLBDEM.BGKCollision(num_components,fluid_bc.group)
# collision_engine.unit_conversion(lb_field)


# collision_engine=openLBDEM.TRTCollision(num_components,,fluid_bc.group,Magic)
# collision_engine.unit_conversion(lb_field,Magic)

collision_engine=openLBDEM.MRTCollision(num_components,fluid_bc.group)
collision_engine.unit_conversion(lb_field)
#==============================================
post_processing_engine=openLBDEM.PostProcessingEngine(0)
#==============================================



#=============================================
rho_cr=tm.log(2.0)
rho0_liq=2.05
rho1_liq=0.05

rho_liq=ti.Vector([rho0_liq,rho1_liq])

@ti.kernel
def init_hydro():
    lb_field.vel.fill([.0,.0])
    for m in range(fluid_bc.group.count[None]):
        ix,iy=fluid_bc.group.group[m]
        r2=(ix-NX_LB/2)**2+(iy-0)**2
        R2=(25)**2
        if r2<R2:
            lb_field.rho[ix,iy,0]=rho0_liq
            lb_field.rho[ix,iy,1]=rho1_liq
        else:
            lb_field.rho[ix,iy,0]=rho1_liq
            lb_field.rho[ix,iy,1]=rho0_liq
init_hydro()

lb_field.init_LBM(collision_engine,fluid_bc.group)
#==============================================
@ti.func
def fluid_spec( x2: int, y2: int):
    is_solid = y2<0
    return 0 if is_solid else 1

@ti.func
def fluid_density( x: int, y: int,component: int) -> float:
    rho_neighbor=0.0
    # 上下边界为亲流体墙体，左右边界为周期边界
    if  y>lb_field.NY-1:
        if component==0:
            rho_neighbor= rho1_liq
        else:
            rho_neighbor= rho0_liq

    x = x % lb_field.NX
    rho_neighbor=  lb_field.rho[x, y,component]
    return rho_neighbor

fluid_spec_bc=openLBDEM.BoundarySpec(geometry_fn=fluid_spec,value_fn=fluid_density)


gA=-0.0
gB=-0.0
gAB=5.0
g_coh=np.array([[gA,gAB],[gAB,gB]])

adhcof=-0.9
gadh=np.array([adhcof,-adhcof])

sc_engine=openLBDEM.ShanChenForceC2(lb_field,g_coh,gadh,fluid_bc.group,fluid_spec_bc)

#==============================================
def history_density():
    print(f'\rDelta rho= {lb_field.rho[lb_field.NX//2,5,0]-lb_field.rho[lb_field.NX//2,5,1]}', end='')

# ==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field,sc_engine)
    sc_engine.apply(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)

    collision_engine.apply(lb_field)
    boundary_engine.apply_boundary_conditions(lb_field)
    
def post():
    pressure = cm.Blues(post_processing_engine.post_pressure(lb_field))
    vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    img1 = np.concatenate((pressure, vel_img), axis=1)
    img2=cm.YlGn(post_processing_engine.post_MC_pressure(lb_field))
    img=np.concatenate((img1,img2),axis=0)

    # img2=cm.Spectral(post_processing_engine.post_denstiy(lb_field))
    return img


start_time = time.time()
gui = ti.GUI(name, (2*NX_LB,2*NY_LB)) 
showmode=1 #1=while # 0=iterations

if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for j in range(50):
            lbm_solve()
        img=post()
        history_density()
        gui.set_image(img)
        gui.show()
else:
    video_manager = ti.tools.VideoManager(output_dir="./results", framerate=24, automatic_build=False)
    for i in range(10):
        for j in range(1000):
            lbm_solve()
        img=post()
        gui.set_image(img)
        gui.show()
        filename="test-angle-adhmin09-"+'%d' % i
        history_density()
        post_processing_engine.writeVTK(filename,lb_field)
        # savefilename = f'2C_unMix_{i:05d}.png'   # create filename with suffix png
        # gui.show(savefilename)

        video_manager.write_frame(img)


    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=False)
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')
    
    # end_time = time.time()
    # elapsed_time = (end_time - start_time)
    # print({elapsed_time})
