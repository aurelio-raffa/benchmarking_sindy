% PWA regression for the identification of a pick and place machine [1].

% [1] A.L. Juloski, W.P.M.H. Heemels, G. Ferrari-Trecate. Data-based hybrid
%     modelling of the component placement process in pick-and-place machines.
%     Control Engineering Practice, 12(10), 1241–1252, 2014.

% Written by V. Breschi, Prato, July 2020


clear all
close all
clc 

%Fix the seed for the reproducibility of the experiment 
rng(100) 

%% Load data 
load headdata_1v6 % Load dataset

%% Construct training and validation set
ndec=10;           %Decimation factor 
T=24000*2/ndec;   %Length of the training set

%Decimate inputs and outputs
u=-100*decimate(u,ndec);
y=decimate(y,ndec);

%Normalize inputs and outputs
u=(u-mean(u))/std(u);
y=(y-mean(y))/std(y);


%Training set
u_train=u(1:T,:);
y_train=y(1:T,:);

%Validation set
u_val=u(T+1:end,:);
y_val=y(T+1:end,:);
Tv=size(u_val,1); % Length of validation dataset

%% Parameters of the model

%Order of the model
na=2; %Features: Y(t-1),...,Y(t-na)
nb=2; %Features: U(t-1),...,U(t-nb)
opt_model=[na;nb];

%% Construct training feature matrix and output

nin=max(na,nb);  % Number of points that are not used for training

X_train=zeros(T-nin,sum(opt_model)+1); %Feature matrix
Y_train=zeros(T-nin,1);                %Output
     
for ind=nin+1:T
   X_train(ind-nin,:)=[y_train(ind-1:-1:ind-na)' u_train(ind-1:-1:ind-nb)' 1];
   Y_train(ind-nin,1)=y_train(ind); 
end

%% Construct feature matrix for validation (one-step-ahead prediction)

X_val=zeros(Tv-nin,sum(opt_model)+1);
for ind=nin+1:Tv
   X_val(ind-nin,:) = [y_val(ind-1:-1:ind-na)' u_val(ind-1:-1:ind-nb)' 1];
end

%% Parameters of the NN

nInputs=size(X_val,2);
nHLayers=1;
nOutputs=1;
batch_s=20;
nNeurons=10;
[fU,fw,sw]=build_ANN(nNeurons,nInputs,nOutputs,nHLayers);
a=sqrt(6/(2*nNeurons));

nClusters=2;

theta_y=zeros(nOutputs,sw,nClusters);
%init=0.1*randn(1,sw,1);
for k=1:nClusters
    theta_y(1,:,k)=.1*randn;%init;
end

%c_type=2; % Like this I'm not using the partition to classify the points
alpha1=0.0015;
beta1=0.9;
beta2=0.999;

max_iter=2;
init_opt=1;

multi_iter=5;

affine=true;

c_type=2;
[theta_y,theta_x,S]=PW_nnid(Y_train',X_train',theta_y,fw,alpha1,beta1,beta2,affine,max_iter,init_opt,batch_s,c_type);
% for idx_ep=1:multi_iter
%     [theta_y,theta_x,S]=PW_nnid(Y_train',X_train',theta_y,fw,alpha1,beta1,beta2,affine,max_iter,init_opt,batch_s,c_type,theta_x);
% end

%% Validate the obtained model

%One-step-ahead prediction
pred=1;
[Y_p,S_p]=validate_pwNN(X_val',theta_y,fw,theta_x(1:end-1,:),theta_x(end,:),pred,opt_model,[],affine);
Y_p=[y_val(1:nin);Y_p];

%Compute the Best Fit Rate indicator
BFR_pred=max(0,1-norm(Y_p(nin+1:end)-y_val(nin+1:end))/norm(y_val(nin+1:end)-mean(y_val(nin+1:end))))*100;
fprintf('BFR in Validation (one-step-ahead prediction): %4.2f\n',BFR_pred)


pred = 2; %Fix prediction type 
[Y_s,S_s]=validate_pwNN(u_val,theta_y,fw,theta_x(1:end-1,:),theta_x(end,:),pred,opt_model,y_val(1:nin),affine);

%Compute the Best Fit Rate indicator
BFR_sim=max(0,1-norm(Y_s(nin+1:end)-y_val(nin+1:end))/norm(y_val(nin+1:end)-mean(y_val(nin+1:end))))*100;
fprintf('BFR in Validation (OL simulation): %4.2f\n',BFR_sim)


linewidth=1.5; %Set the width of plots lines

T_vs = (Tv-nin+1)/(60000/ndec)*15; %Length of the validation set in seconds
time = linspace(0,T_vs,Tv);        %Create time vector for plots

figure
title('One-step-ahead prediction')
subplot(2,1,1)
plot(time(nin+1:end),y_val(nin+1:end),'LineWidth',linewidth);
leg{1}='actual';
hold on
grid on
plot(time(nin+1:end),Y_p(nin+1:end),'r--','LineWidth',linewidth);
leg{2}='predicted';
ylabel('output');  
legend(leg,'Location','NorthWest','Interpreter','LaTeX');

subplot(2,1,2); 
plot(time(nin+1:end),S_p,'k.','LineWidth',linewidth);
grid on
hold on
ylabel('predicted mode'); 
xlabel('time [s]');
ylim([0 3])
%savefig('Pick_and_place_pred.fig')


figure
title('Open-loop simulation')
subplot(2,1,1)
plot(time(nin+1:end),y_val(nin+1:end),'LineWidth',linewidth);
leg{1}='actual';
hold on
grid on
plot(time(nin+1:end),Y_s(nin+1:end),'r--','LineWidth',linewidth);
leg{2}='simulated';
ylabel('output');  
legend(leg,'Location','NorthWest','Interpreter','LaTeX');

subplot(2,1,2); 
plot(time(nin+1:end),S_s(nin+1:end),'k.','LineWidth',linewidth);
grid on
hold on
ylabel('predicted mode'); 
xlabel('time [s]');
ylim([0 3])
%savefig('Pick_and_place_sim.fig')
