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
num_components=2

shear_viscosity=np.array([1e-6,1e-6])
bulk_viscosity=np.array([1e-6,1e-6])
# shear_viscosity=np.array([1e-6,1.5e-5])
# bulk_viscosity=np.array([1e-6,1.5e-5])
rho_l=1#it.fish.get('rho_ball')#1000 # density of fluid
rho_g=1
# rho0=np.sqrt(rho_l*rho_g)
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




#==============================================
name="test"
lb_field=openLBDEM.LBField(name,NX_LB,NY_LB,num_components)
#unit 
lb_field.init_conversion(Cl=Cl,Ct=Ct,Crho=C_rho,shear_viscosity=shear_viscosity,bulk_viscosity=bulk_viscosity)


#==============================================
boundary_engine=openLBDEM.BoundaryEngine()
boundary_classifier=openLBDEM.BoundaryClassifier(NX=NX_LB,NY=NY_LB)
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
MaxNumber=1000
ball_field=openLBDEM.SphereIB(MaxNumber,100)
number=1
idArray=np.array([1])
posArray= np.array([[NX_LB/2,NY_LB/2]])
velArray=np.array([[0.,0.]])
radiusArray = np.array([NY_LB/4.])
RadArray=np.array([0.])
forceArray=np.array([[0.,0.]])
torqueArray=np.array([0.,0.])

ball_field.init_Balls(number,idArray,posArray,velArray,forceArray,torqueArray,radiusArray,RadArray)
ball_field.init_IB_nodes()

boundary_engine.Mask_cricle_identify(lb_field,0.5*NX_LB,0.5*NY_LB,0.25*NX_LB)
#==============================================
rho_cr=tm.log(2.0)
rho0_liq=2.05
rho1_liq=0.05
rho_liq=ti.Vector([rho0_liq,rho1_liq])
# rho_gas=ti.Vector([rho0_gas,rho1_gas])


@ti.func
def smooth_step(r, R, transition_width):
    return 0.5 * (1 - tm.tanh((r - R) / transition_width))

@ti.kernel
def init_hydro():
    lb_field.vel.fill([.0,.0])
    for m in range(fluid_bc.group.count[None]):
        ix,iy=fluid_bc.group.group[m]
        if iy<lb_field.NY/2:
            lb_field.rho[ix,iy,0]=rho0_liq
            lb_field.rho[ix,iy,1]=rho1_liq
        else:
            lb_field.rho[ix,iy,0]=rho1_liq
            lb_field.rho[ix,iy,1]=rho0_liq
init_hydro()

@ti.kernel 
def init_hydro_IB(sphere_field:ti.template()):
    # 初始化球体区域
    for n in range(sphere_field.num[None]):
        center_x = sphere_field.Sphere[n].pos.x
        center_y = sphere_field.Sphere[n].pos.y
        radius = sphere_field.Sphere[n].radius
        radius_sq = radius ** 2

        # 计算球体覆盖的网格范围
        min_ix = int(ti.max(0, center_x - radius))
        max_ix = int(ti.min(lb_field.NX - 1, center_x + radius))
        min_iy = int(ti.max(0, center_y - radius))
        max_iy = int(ti.min(lb_field.NY - 1, center_y + radius))

        # 遍历球体覆盖的网格区域
        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                if (ix - center_x) ** 2 + (iy - center_y) ** 2 <= radius_sq:
                    lb_field.vel[ix,iy]=[.0,.0]
    print("init hydro IB")
init_hydro_IB(ball_field)
lb_field.init_LBM(collision_engine,fluid_bc.group)

#==============================================
@ti.func
def fluid_spec( x2: int, y2: int):
    is_solid = (0 <= x2 < lb_field.NX) and (0 <= y2 < lb_field.NY) and (lb_field.mask[x2, y2] == -1)
    return 0 if is_solid else 1

@ti.func
def fluid_density( x: int, y: int,component: int) -> float:
    rho_neighbor=0.0
    # 上下边界为亲流体墙体，左右边界为周期边界
    if y<0:
        rho_neighbor= rho0_liq
    elif y>lb_field.NY-1:
        rho_neighbor= rho1_liq

    x = x % lb_field.NX
    rho_neighbor=  lb_field.rho[x, y,component]
    return rho_neighbor




fluid_spec_bc=openLBDEM.BoundarySpec(geometry_fn=fluid_spec,value_fn=fluid_density)


gA=-0.0
gB=-0.0
gAB=5.0
g_coh=np.array([[gA,gAB],[gAB,gB]])

adhcof=-0.5
gadh=np.array([adhcof,-adhcof])

sc_engine=openLBDEM.ShanChenForceC2(lb_field,g_coh,gadh,fluid_bc.group,fluid_spec_bc)

#==============================================
MDFIteration=3
ib_engine=openLBDEM.IBEngine(MDFIteration)
#==============================================
ibdem_engine=openLBDEM.IBDEMCouplerEngine(lb_field)
ibdem_engine.SphereIBUpdate(ball_field)

post_processing_engine=openLBDEM.PostProcessingEngine(0)


# ==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field,sc_engine)
    sc_engine.apply(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)
    
    ib_engine.MultiDirectForcingLoop(ball_field,lb_field)
    ib_engine.ComputeParticleForce(ball_field)

    collision_engine.apply(lb_field)
    boundary_engine.apply_boundary_conditions(lb_field)
    
def post():
    # pressure = cm.Blues(post_processing_engine.post_pressure(lb_field))
    # vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    # img1 = np.concatenate((pressure, vel_img), axis=1)
    # img2=cm.RdYlBu(post_processing_engine.post_MC_pressure(lb_field))
    # img=np.concatenate((img1,img2),axis=0)

    img=cm.Blues(post_processing_engine.post_denstiy(lb_field))
    return img




post_processing_engine.show()

showmode=1 #1=while # 0=iterations
start_time = time.time()
gui = ti.GUI(name, (1*NX_LB,1*NY_LB)) 


if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for i in range(1):
            for j in range(50):
                lbm_solve()
            img=post()
            gui.set_image(img)
            gui.show()
        print(f'\rDelta p= {lb_field.total_pressure[100,100]-lb_field.total_pressure[0,100]}', end='')

else:
    video_manager = ti.tools.VideoManager(output_dir="./results", framerate=24, automatic_build=False)
    for i in range(10):
        for j in range(1000):
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


# 在计算结束后显示绘图窗口
if showmode == 2:
    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示绘图窗口并进入事件循环