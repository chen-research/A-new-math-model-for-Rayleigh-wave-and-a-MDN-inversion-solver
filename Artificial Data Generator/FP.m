function [C,SF]=FP(T,h,mu1,rho,lamda,n)
%the Forward Problem --  Rayleigh waves dispersion function -- By Yang J.2021
% T-period
% h-thickness
% mu,lamda-Lame coefficients
% rho-density
% n-number layers
% C-Phase velocity
% SF-Secular function related to phase velocity
ii=sqrt(-1);
A=zeros(4,4);
B=zeros(4,1);
QX=zeros(4,1);
N=length(n);
H=20;

 for l1=1:length(n)
    vs(l1)=sqrt(mu1(l1)/rho(l1));
    vp(l1)=sqrt((lamda(l1)+2*mu1(l1))/rho(l1));
 end


f=1/T;
omega=2*pi*f;
c0=0.8*min(vs);
cd=1*max(vs);
dc=1.0/500;
nn=1;
 %%
    while c0<cd
    gamma=omega/c0;
    r=gamma;
    k=gamma;
    

lambda=lamda';
mu=mu1';
lambda=lamda';
for i=1:N
Kp(i)=omega/vp(i);
Ks(i)=omega/vs(i);
ITA1(i)=-(gamma^2-Ks(i)^2)^(1/2);
ITA2(i)=-(gamma^2-Kp(i)^2)^(1/2);
eta1(i)=-(gamma^2-Ks(i)^2)^(1/2);
eta2(i)=-(gamma^2-Kp(i)^2)^(1/2);
end
Z1(N) =-(mu(N)*ITA2(N)*(ITA1(N)^2-gamma^2)/(ITA1(N)*ITA2(N)-gamma^2));
Z2(N)=ii*mu(N)*gamma-(ii*gamma*mu(N)*(ITA1(N)^2-ITA1(N)*ITA2(N)))/(ITA1(N)*ITA2(N)-gamma^2);
Z3(N)=(lambda(N)+2*mu(N))*ii*gamma*(ITA2(N)^2-ITA1(N)*ITA2(N))/(ITA1(N)*ITA2(N)-gamma^2)+ii*gamma*lambda(N);
Z4(N)=((lambda(N)+2*mu(N))*ITA1(N)*(gamma^2-ITA2(N)^2)/(ITA1(N)*ITA2(N)-gamma^2));
%%
for i=N-1:-1:1
a11=-(2*ii)*mu(i)*gamma*eta2(i)+ii*Z1(i+1)*gamma+Z2(i+1)*eta2(i);
a12=-Z1(i+1)*eta1(i)+ii*Z2(i+1)*gamma-mu(i)*(-eta1(i)^2-gamma^2);
a13=(2*ii)*mu(i)*gamma*eta2(i)+ii*Z1(i+1)*gamma-Z2(i+1)*eta2(i);
a14=(Z1(i+1)*eta1(i)+ii*Z2(i+1)*gamma-mu(i)*(-eta1(i)^2-gamma^2));
a21=(ii*Z3(i+1)*gamma+Z4(i+1)*eta2(i)-(lambda(i)+2*mu(i))*eta2(i)^2+lambda(i)*gamma^2);
a22= (-Z3(i+1)*eta1(i)+ii*Z4(i+1)*gamma-ii*(lambda(i)+2*mu(i))*gamma*eta1(i)+ii*lambda(i)*gamma*eta1(i));
a23=(ii*Z3(i+1)*gamma-Z4(i+1)*eta2(i)-(lambda(i)+2*mu(i))*eta2(i)^2+lambda(i)*gamma^2);
a24=(Z3(i+1)*eta1(i)+ii*Z4(i+1)*gamma+ii*(lambda(i)+2*mu(i))*gamma*eta1(i)-ii*lambda(i)*gamma*eta1(i));
a31= ii*gamma*exp(eta2(i)*h(i))  ;
a32= -eta1(i)*exp(eta1(i)*h(i));
a33=ii*gamma*exp(-eta2(i)*h(i));
a34=eta1(i)*exp(-eta1(i)*h(i));
a41= eta2(i)*exp(eta2(i)*h(i)) ;
a42=ii*gamma*exp(eta1(i)*h(i));
a43= -eta2(i)*exp(-eta2(i)*h(i)) ;
a44=ii*gamma*exp(-eta1(i)*h(i));
Q1=mu(i)*(-gamma^2*exp(eta1(i)*h(i))-eta1(i)^2*exp(eta1(i)*h(i)));
Q2=(2*ii)*mu(i)*gamma*eta2(i)*exp(eta2(i)*h(i));
Q3=mu(i)*(-exp(-eta1(i)*h(i))*eta1(i)^2-exp(-eta1(i)*h(i))*gamma^2);
Q4=-(2*ii)*mu(i)*gamma*exp(-eta2(i)*h(i))*eta2(i);
P1=(ii*(lambda(i)+2*mu(i))*gamma*exp(eta1(i)*h(i))*eta1(i)-ii*lambda(i)*gamma*exp(eta1(i)*h(i))*eta1(i));
P2=((lambda(i)+2*mu(i))*eta2(i)^2*exp(eta2(i)*h(i))-lambda(i)*gamma^2*exp(eta2(i)*h(i)));
P3= (-ii*(lambda(i)+2*mu(i))*gamma*eta1(i)*exp(-eta1(i)*h(i))+ii*lambda(i)*gamma*eta1(i)*exp(-eta1(i)*h(i)));
P4=((lambda(i)+2*mu(i))*exp(-eta2(i)*h(i))*eta2(i)^2-lambda(i)*gamma^2*exp(-eta2(i)*h(i)));
X=a11*a22*a33*a44-a11*a22*a34*a43-a11*a23*a32*a44+a11*a23*a34*a42+a11*a24*a32*a43-a11*a24*a33*a42-...
    a12*a21*a33*a44+a12*a21*a34*a43+a12*a23*a31*a44-a12*a23*a34*a41-a12*a24*a31*a43+a12*a24*a33*a41+...
    a13*a21*a32*a44-a13*a21*a34*a42-a13*a22*a31*a44+a13*a22*a34*a41+a13*a24*a31*a42-a13*a24*a32*a41-...
    a14*a21*a32*a43+a14*a21*a33*a42+a14*a22*a31*a43-a14*a22*a33*a41-a14*a23*a31*a42+a14*a23*a32*a41;
Z1(i)=((-a11*a23*a44+a11*a24*a43+a13*a21*a44-a13*a24*a41-a14*a21*a43+a14*a23*a41)*Q1+...
    (a12*a23*a44-a12*a24*a43-a13*a22*a44+a13*a24*a42+a14*a22*a43-a14*a23*a42)*Q2+...
    (-a11*a22*a43+a11*a23*a42+a12*a21*a43-a12*a23*a41-a13*a21*a42+a13*a22*a41)*Q3+...
    (a11*a22*a44-a11*a24*a42-a12*a21*a44+a12*a24*a41+a14*a21*a42-a14*a22*a41)*Q4)/X;
Z2(i)=((a11*a23*a34-a11*a24*a33-a13*a21*a34+a13*a24*a31+a14*a21*a33-a14*a23*a31)*Q1+...
    (-a12*a23*a34+a12*a24*a33+a13*a22*a34-a13*a24*a32-a14*a22*a33+a14*a23*a32)*Q2+...
    (a11*a22*a33-a11*a23*a32-a12*a21*a33+a12*a23*a31+a13*a21*a32-a13*a22*a31)*Q3+...
    (-a11*a22*a34+a11*a24*a32+a12*a21*a34-a12*a24*a31-a14*a21*a32+a14*a22*a31)*Q4)/X;
Z3(i)=((-a11*a23*a44+a11*a24*a43+a13*a21*a44-a13*a24*a41-a14*a21*a43+a14*a23*a41)*P1+...
    (a12*a23*a44-a12*a24*a43-a13*a22*a44+a13*a24*a42+a14*a22*a43-a14*a23*a42)*P2+...
    (-a11*a22*a43+a11*a23*a42+a12*a21*a43-a12*a23*a41-a13*a21*a42+a13*a22*a41)*P3+...
    (a11*a22*a44-a11*a24*a42-a12*a21*a44+a12*a24*a41+a14*a21*a42-a14*a22*a41)*P4)/X;
Z4(i)=((a11*a23*a34-a11*a24*a33-a13*a21*a34+a13*a24*a31+a14*a21*a33-a14*a23*a31)*P1+...
    (-a12*a23*a34+a12*a24*a33+a13*a22*a34-a13*a24*a32-a14*a22*a33+a14*a23*a32)*P2+...
    (a11*a22*a33-a11*a23*a32-a12*a21*a33+a12*a23*a31+a13*a21*a32-a13*a22*a31)*P3+...
    (-a11*a22*a34+a11*a24*a32+a12*a21*a34-a12*a24*a31-a14*a21*a32+a14*a22*a31)*P4)/X;

end
%%
F=Z1(1)*Z4(1)-Z2(1)*Z3(1);
secularfunr(nn)=abs((real(F)));   
    c(nn)=c0;
    nn=nn+1;
    c0=c0+dc;
    end
    C=c';
    SF=secularfunr';
