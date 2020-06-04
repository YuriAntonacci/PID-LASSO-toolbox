%% Conditional Information Transfer Estimation - theoretical example -
% analysis of simulated 5-variate VAR process

clear; close all; clc;

load('TimeSeries.mat')

%%% MVAR process parameters

M=size(Am,1);
Su=eye(M);
p=size(Am,2)/M;
N=100; % Number of data samples (set the desired)
kratio=(N*M)/(M*M*p);
Y=Y(1:N,:);

%% Theoretical cTE network

%%% ISS paramters
[A,C,K,V,Vy] = varma2iss(Am,[],Su,eye(M));

% % Conditional Tranfer Entropy (eq. 9)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Ti_js(jj,ii)=0.5*log(round(Sj_js,15)/round(Sj_ijs,15));
            
        end
    end
end
THEO=Ti_js;

%% cTE network - OLS -

% MVAR model identification
[Am_OLS,Su_OLS,Yp_OLS,Up_OLS,Z_OLS,Yb_OLS]=idMVAR(Y',p,0);

%%% ISS paramters
[A,C,K,V,Vy] = varma2iss(Am_OLS,[],Su_OLS,eye(M));

% % Conditional Tranfer Entropy (eq. 9)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Ti_js(jj,ii)=0.5*log(round(Sj_js,15)/round(Sj_ijs,15));
            
        end
    end
end
cTE=Ti_js;

% testing significance for the estimated cTE values with surrogates 
% See Sect. 2.4 for a description

[Ti_jsSurr]=cTEsurrogate(Y,100,p);
thr=prctile(Ti_jsSurr,95,3);
cTE(cTE<=thr)=0;
OLS=cTE;

%% cTE network - LASSO - 

%%% LASSO paramters
lambda=logspace(-2,1,100); % interval of lambdas
folds=10; %number of folds

% MVAR model identification
[lopt,GCV,df,Am_LASSO,Su_LASSO] = SparseId_MVAR(Y,p,lambda,folds);

%%% plot of GCV function
Fig1=figure;
% set(Fig1(1),'Position',[196   469   560   418]);
plot(log10(lambda),GCV,'LineWidth',1.3)
xlabel('log( {\lambda} )');
ylabel('log (GCV)')
hold on
[ind]=find(lambda==lopt);
plot(log10(lopt),GCV(ind),'or','LineWidth',1.7)
tit=sprintf('Selected lambda = %s',num2str(lopt));
title(tit);

% ISS parameters
[A,C,K,V,Vy] = varma2iss(Am_LASSO,[],Su_LASSO,eye(M)); %


% % Conditional Tranfer Entropy (eq. 9)

for jj=1:M
    for ii=1:M
        if ii~=jj
            ss=1:M;  ss(ismember(ss,[ii,jj]))=[];  % all processes\ (i,j)
            tmp=iss_PCOV(A,C,K,V,[jj ss]);
            Sj_js=tmp(1,1);
            tmp=iss_PCOV(A,C,K,V,[jj ii ss]);
            Sj_ijs=tmp(1,1);
            Ti_js(jj,ii)=0.5*log(round(Sj_js,15)/round(Sj_ijs,15));

        end
    end
end
LASSO=Ti_js;

%% plot of cTE networks

figure
subplot(1,3,1);
plot_pw(THEO);
title('Theo');
subplot(1,3,2);
plot_pw(OLS);
title('OLS');
subplot(1,3,3);
plot_pw(LASSO);
title('LASSO')
tit=sprintf('Samples=%s, Kratio=%s',num2str(N),num2str(kratio));
suptitle(tit)