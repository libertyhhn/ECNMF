function [Vr,tElapsed,object] = main_ECNMF( data,mu,parameter)
% object: %  1/p *||X^p-U^pV^p||2+beta(||W'Vlab-Y||2+gamma||W||21)+ mu*tr(V^pL^p(V^p)')
% s.t. X,U,V>=0.
% V=[V1,...V_P;Vc]=[[Vsl,Vsul];  Vlab=V(:,1:nl)

% other input: labPer ！！ The percentage of label available during training, i.e., 0.1:0.1:0.5
%              nctPer ！！ partially shared ratio       
% output: Vr ！！  for GPSNMF^k ;  
%         Vw ！！  for GPSNMF^w
%         testLabel ！！ true label for test

%% allocation
numOfTypes = size(data,2);     % 篇叔方
numOfDatas = size(data{1},2);  % 劔云方
parameter.mu=mu;

%% Construct graph Laplacian
nSmp=size(data{1},2);
if mu>0
for v = 1:length(data)
    parameter.WeightMode='Binary';
    W{v}=constructW_cai(data{v}',parameter); 
    if mu > 0
        W{v} = mu*W{v};
        DCol = full(sum(W{v},2));
        D{v} = spdiags(DCol,0,nSmp,nSmp);
        L{v} = D{v} - W{v};
        if isfield(parameter,'NormW') && parameter.NormW
            D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;
            L{v} = D_mhalf*L{v}*D_mhalf;
        end
    end
end
else
    for v = 1:length(data)   
        L{v}=0;  
        D{v}=0;
        W{v}=0;
    end  
end
 parameter.L=L;   
 parameter.D=D;
 parameter.S=W;
 n = parameter.nwf/2;
 
%  for i = 1:numOfTypes
% %      [U{i}, V{i}] = NNDSVD(data{i}, parameter.nwf, 0);
%      [Utemp, Vtemp] = NNDSVD(data{i}, n, 0);
%      V{i} = [Vtemp;Vtemp];
%      U{i} = [Utemp,Utemp];
%  end
%% do GPSNMF
[Vr, U, V, numIter,tElapsed,object] = ECNMF(data,parameter);
% [Vr, U, V, numIter,tElapsed,object] = AO_CPSNMF_2(data,parameter);
%% object curve
% figure;
% plot(object);
end