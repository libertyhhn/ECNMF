function [Vr, U, V, numIter,tElapsed,object] = ECNMF(X, option, U, V)

tStart=tic;

nv = length(X); % number of views

optionDefault.maxIter = 200;  %  
optionDefault.minIter = 10;   %  
optionDefault.dis = 1;
optionDefault.epsilon=1e-6;
 
optionDefault.lambda2 = 0.1; % 
optionDefault.alpha = 1/nv*ones(1,nv); % 
optionDefault.nctPer = 0.5; % 
optionDefault.nwf = 40; % 
optionDefault.isUNorm =1; %  
optionDefault.isXNorm =1; %  
optionDefault.isSqrt =1; % 
optionDefault.typeB = 1; % 
optionDefault.gamma = 0.1; % Consistency 
if ~exist('option','var')
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);  %  
end 

nwf = option.nwf;
lambda = option.lambda2;
gamma = option.gamma;
alpha = option.alpha;
nctPer = option.nctPer;
isUNorm = option.isUNorm;
isXNorm = option.isXNorm;
maxIter = option.maxIter;
minIter = option.minIter;
epsilon = option.epsilon;
dis = option.dis;
isSqrt = option.isSqrt;
typeB = option.typeB;
mu=option.mu;
L=option.L;
DD=option.D;
S=option.S;
c = option.numOfCluster;

if isXNorm % 
    for i = 1:nv
        X{i} = normalize(X{i},0,1);
    end
end

nf = zeros(1,nv);

for i = 1:nv
    nf(i) = size(X{i},1); %number of i-view feature
end

ns = size(X{1},2);   %  
% nst = round(nwf/(nctPer/(1-nctPer)+nv)); % number of specific factor 
 
nst = nwf/2;
nct = nwf/2;
nt = nst+nct;

VC = zeros(nct,ns);
% init beta and W
beta = ones(nv,1).*(1/nv);
W = cell(1,nv);
for v_ind = 1:nv
    W{1,v_ind} = eye(nct);
end

if ~exist('V','var')        
    for i = 1:nv
        Vs{i} = rand(nst,ns); 
        Vc{i} = rand(nct,ns); 
        V{i} = [Vs{i};Vc{i}];  
    end
else  
    for i = 1:nv
        Vc{i} = V{i}(1+nst:end,:);
        Vs{i} = V{i}(1:nst,:);       
    end
end

if ~exist('U','var')
    for i = 1:nv
    	U{i} = rand(nf(i),nt); 
        if isUNorm %             
            U{i} = normalize(U{i},0,0); %  
        end
    end
end

for i = 1:nv     % 
	Us{i} = U{i}(:,1:nst);  
	Uc{i} = U{i}(:,1+nst:end); 
	obj{i}=[];
end

objY=[];
object=[];

for j=1:maxIter  
    %% update HC
    HC = zeros(ns,nct);
    for v_ind = 1:nv
        HC = HC + beta(v_ind) * Vc{v_ind}' * W{v_ind};
    end
    [Uh1,~,Vh1] = svd(HC,'econ');
    VC = Vh1 * Uh1';

   	%% update U
    for i = 1:nv
        U{i} = U{i}.*(max(X{i}*V{i}',eps)./max(U{i}*(V{i}*V{i}'),eps));
        if isUNorm % 
            U{i} = normalize(U{i},0,0);
        end
        Us{i} = U{i}(:,1:nst); 
        Uc{i} = U{i}(:,1+nst:end); 
    end
    for i = 1:nv
            WVC = W{i}* VC;
    WVCp = (abs(WVC)+WVC)./2;
    WVCn = (abs(WVC)-WVC)./2;  
        %  
        if isSqrt
        %Exclu Vc
            Rab = 0;         
            for k=1:nv
                Rab = Rab + Vs{k};
            end  
        %Exclu Vs
            Rcd = 0;         
            for k=1:nv
                if (k==i)
                    continue;
                end
                Rcd = Rcd + Vs{k};
            end             
        %Consisten Vc
            Ref = 0;         
            for k=1:nv
                if (k==i) 
                    continue;
                end
                Ref = Ref + Vc{k};
            end  
            Rgh = 0;         
            for k=1:nv
                Rgh = Rgh + Vc{k};
            end              
            if mu > 0  
                Vs{i} = Vs{i}.*sqrt(max(alpha(i)*Us{i}'*X{i}+mu*Vs{i}*S{i},eps)...  % 
                    ./max(alpha(i)*Us{i}'*U{i}*V{i}+mu*Vs{i}*DD{i}+lambda*Vc{i},eps));  % 这道式子跟论文不太一样
            else      
                Vs{i} = Vs{i}.*sqrt(max(alpha(i)*Us{i}'*X{i},eps)...
                    ./max(alpha(i)*Us{i}'*U{i}*V{i}+lambda*Vc{i},eps));
            end   
        else
            Vs{i} = Vs{i}.*(max(alpha(i)*Us{i}'*X{i},eps)...
                ./max(alpha(i)*Us{i}'*U{i}*V{i}+lambda*Vc{i},eps));
        end     
        % 计算更新Vc相关值
         if isSqrt
             if mu>0
                Vc{i} = Vc{i}.*sqrt(max(alpha(i)*Uc{i}'*X{i}+mu*Vc{i}*S{i}+gamma*beta(i)*WVCp,eps)...
                    ./max(alpha(i)*Uc{i}'*U{i}*V{i}+mu*Vc{i}*DD{i}+lambda*Vs{i}+gamma*beta(i)*WVCn,eps));
             else
                Vc{i} = Vc{i}.*sqrt(max(alpha(i)*Uc{i}'*X{i}+gamma*beta(i)*WVCp,eps)...
                    ./max(alpha(i)*Uc{i}'*U{i}*V{i}+lambda*Vs{i}+gamma*beta(i)*WVCn,eps));
             end
         else
            Vc{i} = Vc{i}.*(max(alpha(i)*Uc{i}'*X{i}+gamma*beta(i)*WVCp,eps)...
                ./max(alpha(i)*Uc{i}'*U{i}*V{i}+lambda*Vs{i}+gamma*beta(i)*WVCn,eps));
         end            
    end
    
    % update W
    for v_ind = 1:nv
        Q = beta(v_ind) * Vc{v_ind} * VC';
        [Uh2,~,Vh2] = svd(Q,'econ');
        W{v_ind} = Uh2  * Vh2';
    end
    
    % update beta
    f = zeros(nv,1);
    for v_ind = 1:nv
        f(v_ind) = trace(Vc{v_ind}' * W{v_ind} * VC);
    end
    beta = f/sqrt(sum(f.^2));
    
    for i =1:nv
        V{i}=[Vs{i};Vc{i}];
        objRec(i) = alpha(i)*(norm(X{i}-U{i}*V{i},'fro')^2);   
        objGraph(i)=mu*trace(V{i}*L{i}*V{i}');
        for www = 1:nv
            dorm_Consis(www) = trace(VC*beta(www)* Vc{www}' * W{www});
        end   
        objExclu(i)=lambda*trace(Vs{i}*Vc{i}');
        objConsis(i)=sum(dorm_Consis);
    end
    object(j) =sum(objRec)+sum(objGraph)+sum(objExclu)-gamma*sum(objConsis);%+beta*objY(j);
    if j>1
        obj_sub = object(j)-object(j-1);
    end
    if mod(j,5)==0 || j==maxIter
        isStop = isIterStop(obj_sub, epsilon, j, maxIter, dis, minIter);
        if isStop
            break;            
        end
    end
end

Vr=[];
Vcall = zeros(nct,ns);
for i =1:nv
    Vr = [Vr;Vs{i}];
    Vcall = Vcall+Vc{i};
end
Vcall = Vcall./nv;
% Vr = [Vr;Vcall];
Vr = [Vr;VC];

numIter = j;
tElapsed=toc(tStart);
end