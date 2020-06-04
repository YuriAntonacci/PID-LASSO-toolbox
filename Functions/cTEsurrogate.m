%% NULL DISTRIBUTION EVALUATION FOR CONDITIONAL TRANSFER ENTROPY

%%% input:
% Y, N*M matrix of time series
% iter, number of iterations (at least 100)
% p, model order
%%% output,
% Ti_jsSurr, M*M*iter matrix of surrogate values for conditional TE
%
function [Ti_jsSurr]=cTEsurrogate(Y,iter,p)

M=size(Y,2);

for ns=1:iter
    for m=1:M
        Data_surr(:,m,ns)=surriaafft(Y(:,m));
    end
    % MVAR model identification
    [Am_surr,Su_surr]=idMVAR(Data_surr(:,:,ns)',p,0);
    
    %%% ISS paramters
    [A,C,K,V,Vy] = varma2iss(Am_surr,[],Su_surr,eye(M));
    
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
    Ti_jsSurr(:,:,ns)=Ti_js;
    
end