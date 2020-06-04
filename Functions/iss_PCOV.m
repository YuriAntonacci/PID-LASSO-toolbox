function [VR, lambda0] = iss_PCOV(A,C,K,V,ii)

% Compute partial covariances considering only the processes specified by ii

[m,m1]  = size(A); assert(m1 == m);
[n,m1]  = size(C); assert(m1 == m);
[m1,n1] = size(K); assert(n1 == n && m1 == m);
[n1,n2] = size(V); assert(n1 == n && n2 == n);

% i1 = i1(:)'; assert(all(i1 >=1 & i1 <= n));
ii = ii(:)'; assert(all(ii >=1 & ii <= n));

KVSQRT = K*chol(V,'lower');
[~,VR,rep] = ss2iss(A,C(ii,:),KVSQRT*KVSQRT',V(ii,ii),K*V(:,ii)); % reduced model innovations covariance
if rep < 0
    if     rep == -1, warning('DARE: eigenvalues on/near unit circle');
    elseif rep == -2, warning('DARE: couldn''t find stablising solution');
    end
    return
end
if rep > sqrt(eps), warning('DARE: there were accuracy issues (relative residual = %e)',rep);
    return
end

% determine the variance of the process lambda0=E[Yn Yn']
O=dlyap(A,K*V*K');
lambda0=C*O*C'+V;

