clear; clc;
%% Define optimization parameter and range
maxEpochs = optimizableVariable('maxEpochs',[100,800],'Type','integer');        % Maximum number of training epochs
miniBatchSize = optimizableVariable('miniBatchSize',[5,50],'Type','integer');   % Mini batch size
alpha0 = optimizableVariable('alpha0',[0.0001,0.001],'Type','real');            % InitialLearnRate
tau = optimizableVariable('tau',[10,100],'Type','integer');                     % LearnRateDropPeriod
gamma = optimizableVariable('gamma',[0.9,0.99],'Type','real');                  % LearnRateDropFactor
gradientThreshold = optimizableVariable('gradientThreshold',[0.9,1.1],'Type','real');       % LearnRateDropFactor
nTrial = 65;
dataset = load("RVE_all_data.mat");
load("gru500Net.mat", 'net');
%% Settings
display = 'none';
folderPath = fullfile(pwd,"ParameterTuning_Finetune");
baseFileName = 'trial';

%% Optimization
FinalValidationLoss = @(params) trainNetworkAndReturnValidationLoss_Finetune(params,dataset,net,folderPath,baseFileName,display);
optimizationResults = bayesopt( ...
    FinalValidationLoss, ...
    [maxEpochs,miniBatchSize,alpha0,tau,gamma,gradientThreshold], ...
    "MaxObjectiveEvaluations",nTrial);

%% Rename best network
[~,bestIteration] = min(optimizationResults.ObjectiveMinimumTrace);
oldName = fullfile(folderPath, append("trial_",int2str(bestIteration),".mat"));
newName = fullfile(folderPath, append("trial_",int2str(bestIteration),"_best.mat"));
movefile(oldName, newName);

%% Save settings
tempPath = fullfile(folderPath, "experiment_setting.mat");
save(tempPath,"optimizationResults")
