clc;clear
R=20:0.1:100;
gamma=0.1621;
deltap=gamma./R;
hold on
plot(1./R,deltap,DisplayName='gamma=0.1621',LineWidth=1)



data=[1/75  0.0020017027854919434
    1/65  0.00236088037490844735
    1/50 0.00318;
    1/25 0.0066182613372802734];

scatter(data(:,1), data(:,2),DisplayName='Gab=6.0',MarkerFaceColor='auto')



x=data(:,1)
y=data(:,2)
legend('Location','northwest')
box on

xlabel('\Delta x /R')
ylabel('\Delta p')

