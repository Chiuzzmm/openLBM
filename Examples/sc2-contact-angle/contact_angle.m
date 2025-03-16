clc;clear
color_bule=["#ADD5F7","#7FB2F0","#4E7AC7","#35478C","#16193B"];
color_orange=["#FFCA50","#FC9D1C","#EB6528","#9E332E","#4D1B2F"];
color_green=["#02A676","#008C72","#007369","#005A5B","#003840"];
color_gray=["#BABABA","#969696","#707070","#3B3B3B","#2B2B2B"];
color_pink=["#C07DB0","#DBADD2","#F2DDEE","#9250BC","#7E55A3"];
color_purple=["#F2DDEE","#DBADD2","#C07DB0","#9250BC","#7E55A3"];
hold on 

Gadh2=-1:0.01:1;
Gadh1=-Gadh2

Gcoh=2.5
delta_rho=2
theta=acos((Gadh2-Gadh1)./(Gcoh.*((delta_rho)/2))) *180/pi;
plot(Gadh1,theta,Color=color_orange(3))


%%

adh=[
    0.0;
    0.5;
    0.9;]

% data=[H,L]
data=[
    41 ,83;
    47,62;
    52,45;]

H=data(:,1)
L=data(:,2)
R=(4*H.^2+L.^2)./8./H;
theta_sim=atan((H-R)*2./L)*180/pi+90;
scatter(adh,theta_sim,MarkerEdgeColor=color_bule(3))

adh=[
    -0.9;
    -0.5;
    ]
data=[
    27,137;
    34,107;
   ]
H=data(:,1)
L=data(:,2)
R=(4*H.^2+L.^2)./8./H;
theta_sim=90-atan(-(H-R)*2./L)*180/pi
scatter(adh,theta_sim,MarkerEdgeColor=color_bule(3))


xlabel("$G_{\mathrm{adh} ,1}$")
ylabel("contact angle, $^\circ $")