function [num_workers, isexist_ParallelToolBox]  = getWorkersAvailable()
% First checks if Parallel Computing Toolbox exists and then
% Returns num of workers available for executing parallel computing
%
available_toolboxes  = ver;
isexist_ParallelToolBox =  false;
num_workers  = 0; % default is set to 0 = single core processing
index =1;
while index < size(available_toolboxes,2)
    pack='Parallel Computing Toolbox';
    
    if strcmp(available_toolboxes(index).Name,pack)
        isexist_ParallelToolBox =  true;
        break;
    end
    
    index  = index + 1;
    
end
% if Paralle Computing toolbox available then get num_workers/CPU cores
% available for batch processing
if isexist_ParallelToolBox
    myCluster   = parcluster('local');
    num_workers = myCluster.NumWorkers;
end