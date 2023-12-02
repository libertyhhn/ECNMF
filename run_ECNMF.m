addpath('./print');
addpath('./graph');
addpath('./misc');
addpath('./data');
clear all;

%% parameters setting
parameter.maxIter = 200; % 200

dataname = {'Handwritten10_6_2k'};
numdata = length(dataname);

mu=0.1;  % 0.1 graph
parameter.lambda2=0.0001; % 
parameter.gamma=0.0001;   % 

for cdata = 1:numdata
%% read data
idata = cdata;
datadir = 'data/';
dataset = char(dataname(idata));
dataf = [datadir, cell2mat(dataname(idata))];
load(dataf);

%% normalize data and graph
nSmp=size(data{1},2);
C=length(unique(label));
parameter.nwf = 20;    %  
parameter.numOfCluster = C;
for v = 1:length(data)
    data{v} = mapminmax(data{v},0,1); % 
end


%% rand_seed
for i =  1:3
    parameter.rand_num=i;    
    [Vr,t,object] = main_ECNMF( data,mu,parameter);
   [ac1(i), nmi1(i), Pri1(i),AR1(i),F1(i),P1(i),R1(i)] = printResult(Vr', label', C, 1);
end
ac1m = mean(ac1); nmi1m = mean(nmi1); Pri1m = mean(Pri1); AR1m = mean(AR1); F1m = mean(F1); P1m = mean(P1); R1m = mean(R1);
ac1s = std(ac1); nmi1s = std(nmi1); Pri1s = std(Pri1); AR1s= std(AR1);F1s= std(F1);P1s= std(P1);R1s= std(R1);
fprintf('3times ac: %0.2f\tnmi:%0.2f\tpur:%0.2f\tar:%0.2f\tf_sc:%0.2f\tpre:%0.2f\trec:%0.2f\n', ac1m*100, nmi1m*100, Pri1m*100,AR1m*100,F1m*100,P1m*100,R1m*100);
k = [ac1m,ac1s,nmi1m,nmi1s,Pri1m,Pri1s,AR1m,AR1s,F1m,F1s,P1m,P1s,R1m,R1s];
eva = roundn(k*100,-2);
fname = ['./results_ECNMF/',dataset,'-',num2str(mu),'-',num2str(parameter.nwf),'-',...
    num2str(parameter.lambda2),'-',num2str(parameter.gamma),'.mat' ];
save(fname,'eva','Vr','t','object');
Tname = ['./results_ECNMF/',dataset,'.txt'];
dlmwrite(Tname,[mu, parameter.nwf, parameter.lambda2, parameter.gamma],'-append','delimiter','\t','newline','pc');
dlmwrite(Tname,eva,'-append','delimiter','\t','newline','pc');
end