%% Partial Information Decomposition - theoretical example -
% analysis for theoretical 4-variate VAR processes (eq. 17 of the main document)
clear; close all; clc;

% SIMULATION PARAMETERS
B=0.5; % coupling from Y2 to Y4
C=1; % coupling from Y1 to Y2 and from Y1 to Y3 


%% simulation design
%%% Y1->Y2 with coupling C, Y1->Y3 with coupling C
%%% Y2->Y4 with coupling B, Y3->Y4 with coupling 1-B
M=4; 
jj=4; % index of target 4 or 1 as described in the main document
ii=2; % index of first driver
kk=3; % index of second driver


%%% MVAR process parameters
p=2;
Su1=1;Su2=1;Su3=1; Su4=1;
r1=0.95; f1=0.1; % autonomous oscillation Y1
r2=0.95; f2=0.025; % autonomous oscillation  Y2
r3=0.95; f3=0.025; % autonomous oscillation  Y3

Su=eye(M);% residual covariance matrix
Su(1,1)=Su1; Su(2,2)=Su2;
Su(3,3)=Su3; Su(4,4)=Su4;

Ak=NaN*ones(M,M,p);
%effects at lag 1
Ak(1,:,1)=[2*r1*cos(2*pi*f1) 0 0 0];
Ak(2,:,1)=[C 2*r2*cos(2*pi*f2) 0 0];
Ak(3,:,1)=[C 0 2*r3*cos(2*pi*f3) 0];
Ak(4,:,1)=[0 B 1-B 0];
%effects at lag 2
Ak(1,:,2)=[-r1^2 0 0 0];
Ak(2,:,2)=[0 -r2^2 0 0];
Ak(3,:,2)=[0 0 -r3^2 0];
Ak(4,:,2)=[0 0 0 0];

Am=[];
for kp=1:p
    Am=[Am Ak(:,:,kp)];
end

%% Dataset generation (eqs. 17a, 17b, 17c, 17d)

N=500; % Number of data samples 

kratio=(N*M)/(M*M*p);

U=randn(M,N);
Y=zeros(M,N);
for t=1:N
    if t-p<=0
        Y(1,t)=Y(1,t)+U(1,t);
        Y(2,t)=Y(2,t)+U(2,t);
        Y(3,t)=Y(3,t)+U(3,t);
        Y(4,t)=Y(4,t)+U(4,t);
    else
    Y(1,t)=2*r1*cos(2*pi*f1)*Y(1,t-1)-r1^2*Y(1,t-2)+U(1,t);
    Y(2,t)=2*r2*cos(2*pi*f2)*Y(2,t-1)-r2^2*Y(2,t-2)+Y(1,t-1)+U(2,t);
    Y(3,t)=2*r3*cos(2*pi*f3)*Y(3,t-1)-r3^2*Y(3,t-2)+Y(1,t-1)+U(3,t);
    Y(4,t)=0.5*Y(2,t-1)+0.5*Y(3,t-1)+U(4,t);
    end
end

%% PID Analysis -Theoretical-

% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am,[],Su,eye(M)); %

% Partial Information decomposition
[VR, lambda0] = iss_PCOV(A,C,K,V,jj);
Sj=lambda0(jj,jj);
Sj_j=VR;

tmp = iss_PCOV(A,C,K,V,[jj ii]);
Sj_ji=tmp(1,1);

tmp = iss_PCOV(A,C,K,V,[jj kk]);
Sj_jk=tmp(1,1);

tmp = iss_PCOV(A,C,K,V,[jj ii kk]);
Sj_ijk=tmp(1,1);

% % Partial Information Decomposition

Tik_j=0.5*log(Sj_j/Sj_ijk);  % Joint transfer (i,k)-->j (eq. 12)
Ti_j=0.5*log(Sj_j/Sj_ji);    % Transfer entropy i-->j (eq. 11)
Tk_j=0.5*log(Sj_j/Sj_jk);    % Transfer entropy k-->j (eq. 11) 
Rik_j=min(Ti_j,Tk_j);        % Redundant transfer (MMI PID)
Ui_j=Ti_j-Rik_j;             % Unique transfer (eq. 7)
Uk_j=Tk_j-Rik_j;             % Unique transfer (eq. 8)
Sik_j=Tik_j-Ui_j-Uk_j-Rik_j; % Synergistic transfer (eq. 6)

Theo=[Tik_j,Ti_j,Tk_j,Ui_j,Uk_j,Sik_j,Rik_j];

%% PID Analysis - OLS -

%MVAR model identification
[Am_OLS,Su_OLS,Yp_OLS,Up_OLS,Z_OLS,Yb_OLS]=idMVAR(Y,p,0);

% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am_OLS,[],Su_OLS,eye(M)); %

% Partial Information decomposition
[VR, lambda0] = iss_PCOV(A,C,K,V,jj);
Sj=lambda0(jj,jj);
Sj_j=VR;

tmp = iss_PCOV(A,C,K,V,[jj ii]);
Sj_ji=tmp(1,1);

tmp = iss_PCOV(A,C,K,V,[jj kk]);
Sj_jk=tmp(1,1);

tmp = iss_PCOV(A,C,K,V,[jj ii kk]);
Sj_ijk=tmp(1,1);

% % Partial Information Decomposition

Tik_j=0.5*log(Sj_j/Sj_ijk);  % Joint transfer (i,k)-->j (eq. 12)
Ti_j=0.5*log(Sj_j/Sj_ji);    % Transfer entropy i-->j (eq. 11)
Tk_j=0.5*log(Sj_j/Sj_jk);    % Transfer entropy k-->j (eq. 11) 
Rik_j=min(Ti_j,Tk_j);        % Redundant transfer (MMI PID)
Ui_j=Ti_j-Rik_j;             % Unique transfer (eq. 7)
Uk_j=Tk_j-Rik_j;             % Unique transfer (eq. 8)
Sik_j=Tik_j-Ui_j-Uk_j-Rik_j; % Synergistic transfer (eq. 6)

OLS=[Tik_j,Ti_j,Tk_j,Ui_j,Uk_j,Sik_j,Rik_j];
%% PID Analysis - LASSO - 

% in the main document 300 lambda values and folds=10 
lambda=logspace(-3,0.5,100); 
folds=10; %number of folds

% MVAR Identification
[lopt,GCV,df,Am_LASSO,Su_LASSO] = SparseId_MVAR(Y',p,lambda,folds);

% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am_LASSO,[],Su_LASSO,eye(M)); %

% Partial Information decomposition
[VR, lambda0] = iss_PCOV(A,C,K,V,jj);
Sj=lambda0(jj,jj);
Sj_j=VR;

tmp = iss_PCOV(A,C,K,V,[jj ii]);
Sj_ji=tmp(1,1);

tmp = iss_PCOV(A,C,K,V,[jj kk]);
Sj_jk=tmp(1,1);

tmp = iss_PCOV(A,C,K,V,[jj ii kk]);
Sj_ijk=tmp(1,1);

% % Partial Information Decomposition

Tik_j=0.5*log(Sj_j/Sj_ijk);  % Joint transfer (i,k)-->j (eq. 12)
Ti_j=0.5*log(Sj_j/Sj_ji);    % Transfer entropy i-->j (eq. 11)
Tk_j=0.5*log(Sj_j/Sj_jk);    % Transfer entropy k-->j (eq. 11) 
Rik_j=min(Ti_j,Tk_j);        % Redundant transfer (MMI PID)
Ui_j=Ti_j-Rik_j;             % Unique transfer (eq. 7)
Uk_j=Tk_j-Rik_j;             % Unique transfer (eq. 8)
Sik_j=Tik_j-Ui_j-Uk_j-Rik_j; % Synergistic transfer (eq. 6)

LASSO=[Tik_j,Ti_j,Tk_j,Ui_j,Uk_j,Sik_j,Rik_j];

%% Figure of PID measures
figure
bar([Theo;OLS;LASSO]')
leg={'Theo','OLS','LASSO'};
legend(leg)
set(gca,'XTickLabel',{'Tik-j','Ti-j','Tk-j','Ui-j','Uk-j','Sik-j','Rik-j'},'FontName','TimesNewRoman')
tit=sprintf('Samples=%s, Kratio=%s  Target=%s',num2str(N),num2str(kratio),num2str(jj));
title(tit);
