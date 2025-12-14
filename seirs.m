function ydot=seirs(t,y)
%Project Model SEIRS
%beta = 0.07 0.1, 0.13, 0.17
beta=0.17; %change as fit
gamma=1/14;
b=12/365000;
d=6.8/365000;
mu=3.2/365000;
alpha=1/7;
omega=1/180;
N=y(1)+y(2)+y(3)+y(4);
ydot=[b*N+omega*y(4)-beta*y(1)*y(3)/N-d*y(1);
beta*y(1)*y(3)/N-alpha*y(2)-d*y(2);
alpha*y(2)-gamma*y(3)-y(3)*(d+mu);
gamma*y(3)-y(4)*(omega+d)];