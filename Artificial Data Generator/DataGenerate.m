%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Run this program to generate layered media data samples
%the Forward Problem --  Rayleigh waves dispersion function -- By Yang J.2021
% T - period
% h - thickness
% mu,lamda - Lame coefficients
% rho - density
% n - number layers
% C - Phase velocity
% Vs - S-wave velocity
% Vp - P-wavevelocity
% SF - Secular function related to phase velocity
% NT - Number of periods
% NP - Number of samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc
N=9;
AllData=[];
NT=50;
NN=399.0;
s=linspace(power(0.5,1/NN),power(80,1/NN),NT);
T=s.^NN;
C=0*T';
n=1:N;
NP=6;
C1=zeros(NP,NT);
%%
for i=1:NP
h(i,:)=[4,4,4,4,4,4,4,4];%h取10-50随机
vs(i,1)=3+0.8*rand;%横波速度取3.5-6.5
vs(i,2)=3.1+0.8*rand;
vs(i,3)=3.2+0.75*rand;
vs(i,4)=3.3+0.7*rand;
vs(i,5)=3.8+0.8*rand;
vs(i,6)=3.9+0.8*rand;
vs(i,7)=4+0.75*rand;
vs(i,8)=4.2+0.6*rand;
vs(i,9)=4.6+rand;
vp(i,:)=vs(i,:).*1.732;
rho(i,:)=0.414.*((vp(i,:).*1000).^0.214);
mu(i,:)=vs(i,:).^2.*rho(i,:);
lambda(i,:)=vp(i,:).^2.*rho(i,:)-2.*mu(i,:);
end
%%
parfor i=1:NP
    C=zeros(1,NT);
    for j=1:50
[V,ZuKang]=FP(T(j),h(i,:),mu(i,:),rho(i,:),lambda(i,:),n);
C(j)=V(findMin(ZuKang));
    end
C1(i,:)=C;
AllDataNew =[h(i,:),vs(i,:),mu(i,:),rho(i,:),lambda(i,:),T,C1(i,:)];
AllData = [AllData; AllDataNew];
end
%%
xlswrite('NewAllData_layer9_30000_kuan_yue_shu04_27.xlsx',AllData);
%%
