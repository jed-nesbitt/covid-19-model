%setting initial conditions
y0=[25 0.00001 0.00001 0];
tspan=[00 1500];
%solving the system
for i=10:10:1000
[t,y]=ode45(@seirs,tspan,y0);
end

%Creating plots of S, E, I, and R over time
plot(t,y);
title("SEIRS Model - R_0=2.2") %change as fit
legend("Suseptible","Exposed","Infected","Recovered")
xlabel("days")
ylabel("Population (millions)")
%creating phase planes
%plot(y(:,1),y(:,3))
%title("Phase plane with R_0 = 2.2") %change as fit
%xlabel("Suseptible People (millions)")
%label("Infected People (millions)")
%Creating phase volumes
%plot3(y(:,1),y(:,3),y(:,4))
%title("Phase Volume of COVID-19 with R_0=2.2") %change as fit
%xlabel("Suseptible people (millions)")
%xlim([5 25])
%ylabel("Infected people (millions)")
%ylim([0 4])
%zlabel("Recovered people (millions)")
%zlim([0 20])