%% PID ANALYSIS OF BRAIN-BODY INTERACTIONS IN REST/STRESS STATES
% PID measures estimated on real time series (Section 4 main doc.)

clear; close all; clc;

% Load data
load('TimeSeriesStress.mat')
p=[3 3 3]; % model order for each condition 
M=size(TimeSeriesStress,2);
N=300;
% Defines brain and body soruces
ii=[1 2 3]; % body drivers
kk=[4 5 6 7];% brain drivers
jj=3; % target process 
Tar={'eta','rho','pi','delta','theta','alpha','beta'};
%% PID Analysis - OLS -

for i_cond=1:3
    data_cond=TimeSeriesStress(:,:,i_cond);
    data_cond=zscore(data_cond,0,1);
    
    % MVAR model identification
    [Am_OLS,Su_OLS,Yp_OLS,Up_OLS,Z_OLS,Yb_OLS]=idMVAR(data_cond(1:N,:)',p(i_cond),0);
    
    %%% ISS paramters
    [A,C,K,V,Vy] = varma2iss(Am_OLS,[],Su_OLS,eye(M));
    
    % Partial Information decomposition
    [VR, lambda0] = iss_PCOV(A,C,K,V,jj);
    Sj=lambda0(jj,jj);
    Sj_j=VR;
    
    ii(ismember(ii,jj))=[];
    kk(ismember(kk,jj))=[];
    tmp = iss_PCOV(A,C,K,V,[jj ii]); % ii body driver
    Sj_ji=tmp(1,1);
    
    tmp = iss_PCOV(A,C,K,V,[jj kk]); % kk brain driver
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
    
    OLS(i_cond,:)=[Tik_j,Ti_j,Tk_j,Ui_j,Uk_j,Sik_j,Rik_j];
    
end

%% PID Analysis - LASSO - 

%%% LASSO paramters
lambda=logspace(-1,0.5,80); % interval of lambdas
folds=20; %number of folds


for i_cond=1:3
    data_cond=TimeSeriesStress(:,:,i_cond);
    data_cond=zscore(data_cond,0,1);
    % MVAR model identification
    
    [lopt,GCV,df,Am_LASSO,Su_LASSO] = SparseId_MVAR(data_cond(1:N,:),p(i_cond),lambda,folds);
    %%% ISS paramters
    [A,C,K,V,Vy] = varma2iss(Am_LASSO,[],Su_LASSO,eye(M));
    
    % Partial Information decomposition
    [VR, lambda0] = iss_PCOV(A,C,K,V,jj);
    Sj=lambda0(jj,jj);
    Sj_j=VR;
    
    ii(ismember(ii,jj))=[];
    kk(ismember(kk,jj))=[];
    tmp = iss_PCOV(A,C,K,V,[jj ii]); % ii body driver
    Sj_ji=tmp(1,1);
    
    tmp = iss_PCOV(A,C,K,V,[jj kk]); % kk brain driver
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
    
    LASSO(i_cond,:)=[Tik_j,Ti_j,Tk_j,Ui_j,Uk_j,Sik_j,Rik_j];
    
    
end

%% plot of estimated PID measures

label={'Tik-j','Ti-j','Tk-j','Ui-j','Uk-j','Sik-j','Rik-j'};
% OLS
HCF=figure('units','inches','position',[0 0 11.7 8.3]);
orient(HCF,'landscape')
hold on
for pp=1:7
    subplot(2,7,pp)
    bar([OLS(:,pp)]);
    set(gca,'XTickLabel',{'Rest','Ment','Game'},'FontName','TimesNewRoman')
    title(label{pp})
    hold on
end
subplot(2,7,1)
ylabel('OLS')

% LASSO
PP=[8:14];
for pp=1:7
    subplot(2,7,PP(pp))
    bar([LASSO(:,pp)]);
    set(gca,'XTickLabel',{'Rest','Ment','Game'},'FontName','TimesNewRoman')
    title(label{pp})
    hold on
end
subplot(2,7,8)
ylabel('LASSO')
tit=sprintf('Target = %s',Tar{jj});
suptitle(tit);