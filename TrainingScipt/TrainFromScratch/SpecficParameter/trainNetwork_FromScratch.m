clear; clc; 

%% Setting
% Load data
load('RVE_all_data.mat');

% Define training parameter and network structure
TrainingParameter_FromScratch;
NetworkStructure_FromScratch; 

% Define network name and save path
saveName = "FromScratch";
skipSave = 0;

%% Apply setting
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'Shuffle',shuffle, ...
    'ResetInputNormalization', resetInputNormalization,...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',gradientThreshold, ...
    'InitialLearnRate',alpha0, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',tau, ...
    'LearnRateDropFactor',gamma, ...
    'Verbose',true, ...
    'VerboseFrequency',VBfreq, ...
    'Plots','none', ...
    'ValidationFrequency',VBfreq, ...
    'ValidationData',{X_valid,Y_valid},...
    'OutputNetwork',outputNetwork);

%% Train the network
clear net
[net, info] = trainNetwork(X_train,Y_train,layers,options);

%% Save network, training process and error
save(append(saveName,'.mat'),'net','info','maxEpochs','miniBatchSize','alpha0','tau','gamma','gradientThreshold')
