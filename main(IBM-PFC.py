print('\n' * 100)
import sys
import os

current_directory = os.getcwd()
openLBDEM_path = os.path.join(current_directory, "openLBDEM")
sys.path.append(openLBDEM_path)

import taichi as ti
import openLBDEM
import numpy as np
from matplotlib import cm
import time
import itasca as it
from itasca import ballarray
import math

testname="test"
ti.init(arch=ti.cpu)
start_time = time.time()

# solve sets
loopnum =20
saveInterval =2500

#==============================================
deviation=it.fish.get('deviation')
# domain and grid v
NX_LB = int(it.fish.get('NX_LB'))+1
NY_LB = int(it.fish.get('NY_LB'))+1
domainx=it.fish.get('domainx')
Cl=domainx/(NX_LB-1) #length conversion  (main)The larger the Cl, the larger the time step

# flow boundary condition
Umax=it.fish.get('lnlet_vel')
pressure_lnlet=140

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

# other const
AngToRad=math.pi/180

# IBM settings
NUMIB_py =1.5# ->0 refine ->2 rough
MDFIteration=1
radiusCOF=3

# cycle time step
dt = Ct  # it.timestep()
dt_dem_max=dt
it.command("model mechanical timestep maximum {}".format(dt_dem_max))



#==============================================
name="test"
lb_field=openLBDEM.LBField(name,NX_LB,NY_LB,num_components)

#unit 
lb_field.init_conversion(Cl=Cl,Ct=Ct,Crho=C_rho,shear_viscosity=shear_viscosity,bulk_viscosity=bulk_viscosity)

#==============================================
p2=np.array(it.fish.get('p2'))/Cl-0.5
p4=np.array(it.fish.get('p4'))/Cl-0.5
p6=np.array(it.fish.get('p6'))/Cl-0.5
p8=np.array(it.fish.get('p8'))/Cl-0.5
boundary_engine=openLBDEM.BoundaryEngine()
boundary_engine.Mask_rectangle_identify(lb_field,p6[0],p2[0],p6[1],p2[1])
boundary_engine.Mask_rectangle_identify(lb_field,p4[0],p8[0],p4[1],p8[1])
boundary_engine.boundary_identify(lb_field)
boundary_engine.writing_boundary(lb_field)
boundary_engine.add_boundary_condition(openLBDEM.BounceBackWall(lb_field.wall_boundary,None))
boundary_engine.add_boundary_condition(openLBDEM.VelocityBoundary(lb_field.inlet_boundary,ULB,3))
boundary_engine.add_boundary_condition(openLBDEM.PressureBoundary(lb_field.outlet_boundary,1.0,1))

#==============================================
macroscopic_engine=openLBDEM.MacroscopicEngine()
#==============================================
# collision_engine=openLBDEM.BGKCollision()
# collision_engine.unit_conversion(lb_field)

# collision_engine=openLBDEM.TRTCollision()
# collision_engine.unit_conversion(lb_field,Magic)

collision_engine=openLBDEM.MRTCollision(num_components)
collision_engine.unit_conversion(lb_field)
#==============================================
post_processing_engine=openLBDEM.PostProcessingEngine(0)

#==============================================
MaxNumber=1000
ball_field=openLBDEM.SphereIB(MaxNumber,100)
number=it.ball.count()
idArray=ballarray.ids() 
posArray=ballarray.pos()/Cl
velArray=ballarray.vel()/lb_field.get_Cu()
radiusArray = ballarray.radius()/Cl-radiusCOF
RadArray=ballarray.extra(1)*AngToRad
forceArray=ballarray.force_app()/lb_field.get_Cforce()
torqueArray=ballarray.moment_app()/lb_field.get_Ctorque()

ball_field.init_Balls(number,idArray,posArray,velArray,forceArray,torqueArray,radiusArray,RadArray)
ball_field.init_IB_nodes()
#==============================================
lb_field.init_hydro(ULB,0.0)
lb_field.init_hydro_IB(ball_field)
lb_field.init_LBM(collision_engine)
#==============================================
MDFIteration=3
ib_engine=openLBDEM.IBEngine(MDFIteration)
#==============================================
ibdem_engine=openLBDEM.IBDEMCouplerEngine(lb_field)
ibdem_engine.SphereIBUpdate(ball_field)
#==============================================
# ball info
def GetAndSetBallInfo():
    posArray=ballarray.pos()/Cl-0.5
    velArray=ballarray.vel()/lb_field.get_Cu()
    it.command("@BallRotation")
    RadArray=ballarray.extra(1)*AngToRad
    ibdem_engine.SphereInfoUpdata(ball_field,posArray,velArray,RadArray)
#==============================================
def GetAndSendBallForce():
  forceinfo = -ibdem_engine.GetSphereForce(ball_field)*lb_field.get_Cforce()
  ballarray.set_force_app(forceinfo)
  # force_magnitudes =np.linalg.norm(forceinfo,axis=1)
  # if np.any(force_magnitudes>1e2):
  #     lb_field.writeVTK("error")
  #     print("Force exceeded the allowed limit. Exiting the program.")
  #     sys.exit(1)
  TorqueInfo= ibdem_engine.GetSphereTorque(ball_field)*lb_field.get_Ctorque()
  ballarray.set_moment_app(TorqueInfo)

#==============================================solve & show
def lbm_solve():
    # LBM SOLVE
    macroscopic_engine.density(lb_field)
    macroscopic_engine.pressure(lb_field)
    macroscopic_engine.force_density(lb_field)
    macroscopic_engine.velocity(lb_field)
    
    ib_engine.MultiDirectForcingLoop(ball_field,lb_field)
    ib_engine.ComputeParticleForce(ball_field)
    
    collision_engine.apply(lb_field)
    collision_engine.stream_inside(lb_field)
    
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
savename = 0
print('solve start')

if showmode==1:
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for i in range(1):
            for j in range(25):
                # DEM SOLVE
                it.command("model solve mechanical time {}".format(dt))

                # BALL INFO
                GetAndSetBallInfo()

                # LBM SOLVE
                lbm_solve()

                #FORCE APPLY
                GetAndSendBallForce()

            img=post()
            gui.set_image(img)
            gui.show()


