% generates iterative amplitude adjusted fourier tranform surrogates 
% Schreiber and Schmitz - Physical Review Letters 1996

% y: orginal time series
% nit: number of desired surrogates
% stop: 'spe'-same spectral density, 'dis' same distribution

function ys=surriaafft(y,nit,stop)
warning off
error(nargchk(1,3,nargin));
if nargin < 3, stop='spe'; end 
if nargin < 2, nit=7; end 




%%
[ysorted,yindice]=sort(y);
my=abs(fft(y));

ys=surrshuf(y); %initialization


%% cicle

for i=1:nit
    % step 1:
    faseys=angle(fft(ys));
    fys=my.*(cos(faseys)+j*sin(faseys));
    ys=ifft(fys);ys=real(ys);
    ys=ys-mean(ys);

    % step 2: 
    [yssorted,ysindice]=sort(ys);
    ypermuted=zeros(length(y),1);
    for i=1:length(y)
        ypermuted(ysindice(i))=ysorted(i);
    end
    ys=ypermuted;

end


if stop=='spe'
    faseys=angle(fft(ys));
    fys=my.*(cos(faseys)+j*sin(faseys));
    ys=ifft(fys);ys=real(ys);
    ys=ys-mean(ys);
end




