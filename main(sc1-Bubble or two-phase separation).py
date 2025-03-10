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
num_components=1
shear_viscosity=np.array([1e-6])
bulk_viscosity=np.array([1e-6])
rho0=1

cs=0.578 # sound speed 
Ma=Umax/cs*1.1 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho0/1 # density conversion (main)



#TRT
Magic=np.array([1/4])

#MRT



gA=-5.0
#==============================================
name="test"
lb_field=openLBDEM.LBField(name,NX_LB,NY_LB,num_components)
#unit 
lb_field.init_conversion(Cl=Cl,Ct=Ct,Crho=C_rho,shear_viscosity=shear_viscosity,bulk_viscosity=bulk_viscosity)
lb_field.set_gravity(ti.Vector([-0.5e-5,0.0]))


#==============================================
boundary_engine=openLBDEM.BoundaryEngine()
boundary_classifier=openLBDEM.BoundaryClassifier(NX=NX_LB,NY=NY_LB)


def fluid_boundary(i,j):
    return lb_field.mask[i,j]==1

fluid=openLBDEM.BoundarySpec(geometry_fn=fluid_boundary)
fluid_bc=openLBDEM.FluidBoundary(spec=fluid)
fluid_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("fluid",fluid_bc)

stream_bc=openLBDEM.PeriodicAllBoundary(spec=fluid)
stream_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("stream",stream_bc)

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

# k1=np.array([1.0,1.0])
# epslion=np.array([1.0,1.0])
# collision_engine=openLBDEM.HuangMRTCollision(num_components,k1,epslion)


#=============================================
rho_cr=tm.log(2.0)
rho_liq=1.93244248895799
rho_gas=0.156413030238316

def strategy_fn(component: int, x: int, y: int) -> float:
    rho_neighbor=0.0
    # 检查坐标是否越界
    in_domain = (0 <= x < lb_field.NX) and (0 <= y < lb_field.NY)
    if not in_domain:
        # 周期边界
        x2=(x+lb_field.NX)%lb_field.NX 
        y2=(y+lb_field.NY)%lb_field.NY
        rho_neighbor= lb_field.rho[component, x2, y2]

    elif lb_field.mask[x, y] == 1:
        rho_neighbor=  lb_field.rho[component, x, y]
    else:
        rho_neighbor=  lb_field.rho_solid[x, y]
    return rho_neighbor


sc_engine=openLBDEM.ShanChenForceC1(lb_field,gA,fluid_bc.group,strategy_fn)
#==============================================

@ti.func
def smooth_step(r, R, transition_width):
    return 0.5 * (1 - tm.tanh((r - R) / transition_width))


@ti.kernel
def init_hydro():
    lb_field.vel.fill([.0,.0])
    for m in range(fluid_bc.group.count[None]):
        ix,iy=fluid_bc.group.group[m]

        # R=50**2
        # r=(ix-lb_field.NX/2)**2+(iy-lb_field.NY/2)**2
        # if r<R:
        #     lb_field.rho[0,ix,iy]=rho_liq
        # else:
        #     lb_field.rho[0,ix,iy]=rho_gas

        lb_field.rho[0,ix,iy]=rho_cr+0.1*ti.random()
init_hydro()

lb_field.init_LBM(collision_engine,fluid_bc.group)


#==============================================
post_processing_engine=openLBDEM.PostProcessingEngine(0)


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
    return img1


start_time = time.time()
gui = ti.GUI(name, (1*NX_LB,2*NY_LB)) 
result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)


showmode=1 #1=while # 0=iterations

if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for j in range(10):
            lbm_solve()
        img=post()

        gui.set_image(img)
        gui.show()
else:
    for i in range(50):
        for j in range(100):
            lbm_solve()
        img=post()
        gui.set_image(img)
        gui.show()
        filename="test"+'%d' % i
        # savefilename = f'2C_unMix_{i:05d}.png'   # create filename with suffix png
        # gui.show(savefilename)

        video_manager.write_frame(img)


    print('Exporting .mp4 and .gif videos...')
    video_manager.make_video(gif=True, mp4=False)
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')
    # macroscopic_engine.writeVTK(filename,lb_field)
    # end_time = time.time()
    # elapsed_time = (end_time - start_time)
    # print({elapsed_time})
