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

cs=0.578 # sound speed 
Ma=Umax/cs*0.8 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion


rho_cr_real= 3.5000000000
p_cr_real=0.750000000
T_cr_real= 0.5714285714

rho_cr_LB= 3.5000000000
p_cr_LB=0.750000000
T_cr_LB= 0.5714285714

gcoh=-1.0
vdw_params = {
    'a': 0.184 , 
    'b': 0.095,
    'R': 1, 
    'G':gcoh,
    'rho_cr':rho_cr_LB,
    'pressure_cr': p_cr_LB,
    'temperature_cr':T_cr_LB,
    }

Tr=0.95
rho_liq=5.11604570   #rhog_sat 
rho_gas=2.026552244   
print(f"density ratio={rho_liq/rho_gas}")

#huang MRT pars
k1=np.array([0.15]) #sufrce tension
epslion=np.array([1.0]) #density ratio

showmode=1 #1=while # 0=iterations
#==============================================
name="test"
lb_field=openLBDEM.LBField(name,NX_LB,NY_LB,num_components)
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

boundary_engine.writing_boundary(lb_field)


#=============================================
@ti.func
def fluid_spec( x2: int, y2: int):
    return 1

@ti.func
def fluid_density( x: int, y: int) -> float:
    rho_neighbor=0.0
    x = x % lb_field.NX
    y=y%lb_field.NY
    rho_neighbor=  lb_field.rho[x, y,0]
    return rho_neighbor

@ti.func
def fluid_T( x: int, y: int) -> float:
    x = x % lb_field.NX
    y = y % lb_field.NY
    T_neighbor=  lb_field.T[x, y]
    return T_neighbor

fluid_spec_bc=openLBDEM.BoundarySpec(geometry_fn=fluid_spec,value_fn=fluid_density)
sc_engine = openLBDEM.ShanChenForceC1(
    lb_field=lb_field,
    group=fluid_bc.group,
    psi=openLBDEM.vdW_psi(params=vdw_params),  # 将参数封装到字典中
    g_coh=gcoh,
    fluid_strategy=fluid_spec_bc,
    T_strategy=fluid_T
)

#=============================================
#unit 
LB_params={
    'Cl':Cl,
    'Ct':Ct,
    'C_rho':None,
    'shear_viscosity':shear_viscosity,
    'bulk_viscosity':bulk_viscosity,
    'rho_cr_real':rho_cr_real,
    'pressure_cr_real':p_cr_real,
    'temperature_cr_real':T_cr_real,
    'sc_field':sc_engine,
}
lb_field.init_conversion(LB_params)


#==============================================
macroscopic_engine=openLBDEM.MacroscopicEngine(fluid_bc.group)

#==============================================
collision_engine=openLBDEM.MRTCollision(num_components,fluid_bc.group)
collision_engine.unit_conversion(lb_field)

# collision_engine=openLBDEM.HuangMRTCollision(num_components,fluid_bc.group,k1,epslion,np.array([[vdw_params['G']]]))
# collision_engine.unit_conversion(lb_field)
#==============================================

@ti.kernel
def init_hydro():
    lb_field.vel.fill([.0,.0])
    for m in range(fluid_bc.group.count[None]):
        ix,iy=fluid_bc.group.group[m]

        R=15**2
        r=(ix-lb_field.NX/2)**2+(iy-lb_field.NY/2)**2
        if r<R:
            lb_field.rho[ix,iy,0]=rho_liq
        else:
            lb_field.rho[ix,iy,0]=rho_gas

        # lb_field.rho[ix,iy,0]=rho_cr+0.1*ti.random()

    # for num in range(2):
    #     for m in range(fluid_bc.group.count[None]):
    #         ix,iy=fluid_bc.group.group[m]

    #         ix = ix % lb_field.NX
    #         iy = iy % lb_field.NY
    #         lb_field.rho[ix,iy,0]=1/4*(lb_field.rho[ix-1,iy,0]+lb_field.rho[ix+1,iy,0]+lb_field.rho[ix,iy+1,0]+lb_field.rho[ix,iy-1,0])
    lb_field.T.fill(Tr*T_cr_LB)
init_hydro()

lb_field.init_LBM(collision_engine,fluid_bc.group)


#==============================================
post_processing_engine=openLBDEM.PostProcessingEngine(0)

@ti.kernel
def cool():
    k=0.00002
    T= T_cr_LB * (0.7 + 0.25 * tm.exp(-k * lb_field.time[None]))
    lb_field.T.fill(T)
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
    macroscopic_engine.time_updata(lb_field)
    cool()
    
def post():
    pressure = cm.Blues(post_processing_engine.post_pressure(lb_field))
    vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    img1 = np.concatenate((pressure, vel_img), axis=1)
    return img1


start_time = time.time()
gui = ti.GUI(name, (1*NX_LB,2*NY_LB)) 
result_dir = "./results"
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)


if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for j in range(10):
            lbm_solve()
        img=post()
        print(f"\rdroplet density 0= {lb_field.rho[100, 100, 0]:.3f}, air density 0= {lb_field.rho[0, 100, 0]:.3f}, ratio={lb_field.rho[100, 100, 0]/lb_field.rho[0, 100, 0]:.3f}, Tr={lb_field.T[100, 100]/T_cr_LB:.3f},Time={lb_field.time[None]}", end='')
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
