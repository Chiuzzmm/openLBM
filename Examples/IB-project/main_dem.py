print('\n' * 100)
import numpy as np
import itasca as it
from itasca import ballarray

cfd_link = it.util.p2pLinkClient()
cfd_link.connect("localhost")

#==============================================================================
deviation=it.fish.get('deviation')
NX_LB = int(it.fish.get('NX_LB'))
NY_LB = int(it.fish.get('NY_LB'))
domainx=it.fish.get('domainx')
Umax=it.fish.get('lnlet_vel')


cfd_link.send_data(deviation)
cfd_link.send_data(NX_LB)
cfd_link.send_data(NY_LB)
cfd_link.send_data(domainx)
cfd_link.send_data(Umax)

dt=cfd_link.read_data()
dt_dem_max=cfd_link.read_data()
it.command("model mechanical timestep maximum {}".format(dt_dem_max))



p2=np.array(it.fish.get('p2'))
p4=np.array(it.fish.get('p4'))
p6=np.array(it.fish.get('p6'))
p8=np.array(it.fish.get('p8'))
cfd_link.send_data(p6)
cfd_link.send_data(p2)
cfd_link.send_data(p4)
cfd_link.send_data(p8)

number=it.ball.count()
idArray=ballarray.ids() 
posArray=ballarray.pos()
velArray=ballarray.vel()
radiusArray = ballarray.radius()
RadArray=ballarray.extra(1)
forceArray=ballarray.force_app()
torqueArray=ballarray.moment_app()

cfd_link.send_data(number)
cfd_link.send_data(idArray)
cfd_link.send_data(posArray)
cfd_link.send_data(velArray)
cfd_link.send_data(radiusArray)
cfd_link.send_data(RadArray)
cfd_link.send_data(forceArray)
cfd_link.send_data(torqueArray)

def GetAndSendBallInfo():
    posArray=ballarray.pos()
    cfd_link.send_data(posArray)
    velArray=ballarray.vel()
    cfd_link.send_data(velArray)
    it.command("@BallRotation")
    RadArray=ballarray.extra(1)
    cfd_link.send_data(RadArray)
    
def GetAndSetBallForce():
    forceinfo=cfd_link.read_data()
    ballarray.set_force_app(forceinfo)
    
    TorqueInfo=cfd_link.read_data()
    ballarray.set_moment_app(TorqueInfo)

#C
while True:
    # DEM SOLVE
    deltat = cfd_link.read_data()
    if deltat == 0.0:
        print("solve finished")
        break
    it.command("model solve mechanical time {}".format(dt))

    # GET BALL INFO
    GetAndSendBallInfo()

    # LBM SOLVE
    # ...
    
    #FORCE APPLY
    GetAndSetBallForce()
    
cfd_link.close() 
del cfd_link 