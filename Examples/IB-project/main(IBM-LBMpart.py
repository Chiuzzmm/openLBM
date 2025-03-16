import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "openLBM"))
import openLBDEM
from matplotlib import cm
import numpy as np
import taichi as ti

# from matplotlib import cm
import time
import sys
import math
from itasca import p2pLinkServer


pfc_link = p2pLinkServer()
pfc_link.start()#


testname="test"
ti.init(arch=ti.cpu)
start_time = time.time()
#==============================================
deviation=pfc_link.read_data()
# domain and grid v
NX_LB = pfc_link.read_data()+1
NY_LB = pfc_link.read_data()+1
domainx=pfc_link.read_data()
Cl=domainx/(NX_LB-1) #length conversion  (main)The larger the Cl, the larger the time step



# flow boundary condition
Umax=pfc_link.read_data()
#Umax=2e-2
pressure_lnlet=140


# flow info
num_components=1
shear_viscosity=[1e-6]
bulk_viscosity=[2.5e-6]
rho_flow=1000

cs=0.578 # sound speed
Ma=Umax/cs*1.5 #The larger the Ma, the larger the time step

#conversion coefficient
Ux_LB=Ma*cs# vel of LB
Uy_LB=0.0
ULB=ti.Vector([Ux_LB,Uy_LB])
Cu=Umax/Ux_LB #vel conversion (main)
Ct=Cl/Cu #time conversion
C_rho=rho_flow/1 # density conversion (main)

#TRT
Magic=[1/4]


# other const
AngToRad=math.pi/180

# IBM settings
NUMIB_py =1.5# ->0 refine ->2 rough
MDFIteration=1
radiusCOF=3


# cycle time step
dt = Ct  # it.timestep()
dt_dem_max=dt
pfc_link.send_data(Ct)
pfc_link.send_data(dt_dem_max)


# solve sets
loopnum =20
saveInterval =2500


#==============================================

lb_field=openLBDEM.LBField(testname,NX_LB,NY_LB,num_components)


#unit 
lb_field.init_conversion(Cl=Cl,Ct=Ct,Crho=C_rho,shear_viscosity=shear_viscosity,bulk_viscosity=bulk_viscosity)

#==============================================
boundary_engine=openLBDEM.BoundaryEngine()
boundary_classifier=openLBDEM.BoundaryClassifier(NX=NX_LB,NY=NY_LB)


p6=pfc_link.read_data()/Cl-0.5
p2=pfc_link.read_data()/Cl-0.5
p4=pfc_link.read_data()/Cl-0.5
p8=pfc_link.read_data()/Cl-0.5

boundary_engine=openLBDEM.BoundaryEngine()
boundary_engine.Mask_rectangle_identify(lb_field,p6[0],p2[0],p6[1],p2[1])
boundary_engine.Mask_rectangle_identify(lb_field,p4[0],p8[0],p4[1],p8[1])

def fluid_boundary(i,j):
    return lb_field.mask[i,j]==1

fluid=openLBDEM.BoundarySpec(geometry_fn=fluid_boundary)
fluid_bc=openLBDEM.FluidBoundary(spec=fluid)
fluid_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("fluid",fluid_bc)

def inside_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 :
        for k in ti.static(range(lb_field.NPOP)):
            ix2=i-lb_field.c[k,0]
            iy2=j-lb_field.c[k,1]
            if ix2>0 and ix2<lb_field.NX-1 and iy2>0 and iy2<lb_field.NY-1 and lb_field.mask[ix2,iy2]==1:
                flag=1
    return flag

inside=openLBDEM.BoundarySpec(geometry_fn=inside_boundary)
inside_bc=openLBDEM.InsideBoundary(spec=inside)
inside_bc.precompute(classifier=boundary_classifier)
boundary_engine.add_boundary_condition("inside",inside_bc)

def wall_boundary(i,j):
    flag=0
    if lb_field.mask[i,j]==1 :
        for k in ti.static(range(lb_field.NPOP)):
            ix2=i-lb_field.c[k,0]
            iy2=j-lb_field.c[k,1]
            if ix2<0 or ix2>lb_field.NX-1 or iy2<0 or iy2>lb_field.NY-1 or lb_field.mask[ix2,iy2]==-1:
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
#==============================================
post_processing_engine=openLBDEM.PostProcessingEngine(0)

#==============================================
MaxNumber=1000
ball_field=openLBDEM.SphereIB(MaxNumber,100)
number=pfc_link.read_data()
idArray=pfc_link.read_data()
posArray=pfc_link.read_data()/Cl-0.5
velArray=pfc_link.read_data()/lb_field.get_Cu()
radiusArray = pfc_link.read_data()/Cl-radiusCOF
RadArray=pfc_link.read_data()*AngToRad
forceArray=pfc_link.read_data()/lb_field.get_Cforce()
torqueArray=pfc_link.read_data()/lb_field.get_Ctorque()

ball_field.init_Balls(number,idArray,posArray,velArray,forceArray,torqueArray,radiusArray,RadArray)
ball_field.init_IB_nodes()

#==============================================
@ti.kernel 
def init_hydro(vel:ti.types.vector(2, ti.f32),pressure_lnlet:float):
    if pressure_lnlet==0.0:
        for m in range(fluid_bc.group.count[None]):
            ix,iy=fluid_bc.group.group[m]
            lb_field.vel[ix,iy]=vel
            for component in range(lb_field.num_components[None]):
                lb_field.rho[ix,iy,component]=1.0/ lb_field.num_components[None]
    else:
        rho_inlet=1+pressure_lnlet*3/lb_field.C_pressure
        for m in range(fluid_bc.group.count[None]):
            ix,iy=fluid_bc.group.group[m]
            k=(1.0-rho_inlet)/lb_field.NX
            lb_field.vel[ix,iy]=ti.Vector([.0,.0])
            for component in range(lb_field.num_components[None]):
                lb_field.rho[ix,iy,component]=(k*ix+rho_inlet)/ lb_field.num_components[None]
    print("init hydro")
init_hydro(ULB,0.0)

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
MDFIteration=3
ib_engine=openLBDEM.IBEngine(MDFIteration)
#==============================================
ibdem_engine=openLBDEM.IBDEMCouplerEngine(lb_field)
ibdem_engine.SphereIBUpdate(ball_field)
#==============================================
# ball info
def GetAndSetBallInfo():
    posArray=pfc_link.read_data()/Cl-0.5
    velArray=pfc_link.read_data()/lb_field.get_Cu()
    RadArray=pfc_link.read_data()*AngToRad
    ibdem_engine.SphereInfoUpdata(ball_field,posArray,velArray,RadArray)
#==============================================
def GetAndSendBallForce():
  forceinfo = -ibdem_engine.GetSphereForce(ball_field)*lb_field.get_Cforce()
  pfc_link.send_data(forceinfo)
  # force_magnitudes =np.linalg.norm(forceinfo,axis=1)
  # if np.any(force_magnitudes>1e2):
  #     lb_field.writeVTK("error")
  #     print("Force exceeded the allowed limit. Exiting the program.")
  #     sys.exit(1)
  TorqueInfo= ibdem_engine.GetSphereTorque(ball_field)*lb_field.get_Ctorque()
  pfc_link.send_data(TorqueInfo)

def LBMSolve():
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)

    ib_engine.MultiDirectForcingLoop(ball_field,lb_field)
    ib_engine.ComputeParticleForce(ball_field)

    collision_engine.apply(lb_field)
    boundary_engine.apply_boundary_conditions(lb_field)

def post():
    pressure = cm.Blues(post_processing_engine.post_pressure(lb_field))
    vel_img = cm.plasma(post_processing_engine.post_vel(lb_field))
    img1 = np.concatenate((pressure, vel_img), axis=1)
    return img1
#==============================================solve & show

showmode=1 #1=while # 0=iterations
start_time = time.time()
gui = ti.GUI(testname, (NX_LB,2*NY_LB)) 
savename = 0
print('solve start')

if showmode==1:
    # while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    for i in range(10000):
        for j in range(10):
            # DEM SOLVE
            pfc_link.send_data(dt) # solve interval

            # BALL INFO
            GetAndSetBallInfo()

            # LBM SOLVE
            LBMSolve()

            #FORCE APPLY
            GetAndSendBallForce()

        img=post()
        gui.set_image(img)
        gui.show()


end_time = time.time()
elapsed_time = (end_time - start_time)/60
print({elapsed_time})


pfc_link.send_data(0.0) # solve interval
pfc_link.close()
del pfc_link