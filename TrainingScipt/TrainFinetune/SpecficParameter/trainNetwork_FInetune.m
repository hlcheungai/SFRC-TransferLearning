clear; clc; 

%% Setting
% Load data and pre-trained network
load('RVE_all_data.mat');
load('gru500Net.mat','net');

% Define training parameter and network structure
TrainingParameter_Finetune;
NetworkStructure_Finetune; 

% Define network name and save path
saveName = "Finetuned";
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

%% Transfer infomation and freeze layers
% Copy normalization parameter from pre-trained input layer to new network
layers(1).Mean = net.Layers(1).Mean;
layers(1).StandardDeviation = net.Layers(1).StandardDeviation;

% Transfer pre-trained parameter to the new neural network and freeze them
layerNames = {layers.Name};
oldLayerNames = {net.Layers.Name};

for i = 1:length(layerNames)
    if ismember(layerNames(i), oldLayerNames)
        j = find(strcmp(layerNames(i), oldLayerNames));
        if isa(layers(i), 'nnet.cnn.layer.GRULayer')
            layers(i).InputWeights = net.Layers(j).InputWeights;
            layers(i).RecurrentWeights = net.Layers(j).RecurrentWeights;
            layers(i).Bias = net.Layers(j).Bias;
        elseif isa(layers(i), 'nnet.cnn.layer.FullyConnectedLayer')
            layers(i).Weights = net.Layers(j).Weights;
            layers(i).Bias = net.Layers(j).Bias;
        end
    end
end

for i = 1:length(layerNames)
    if ismember(layerNames(i), freezeLayers)
        if isa(layers(i), 'nnet.cnn.layer.GRULayer')
            layers(i).InputWeightsLearnRateFactor = 0;
            layers(i).RecurrentWeightsLearnRateFactor = 0;
            layers(i).BiasLearnRateFactor = 0;
        elseif isa(layers(i), 'nnet.cnn.layer.FullyConnectedLayer')
            layers(i).WeightLearnRateFactor = 0;
            layers(i).BiasLearnRateFactor = 0;
        end
    end
end
%% Train the network
clear net
[net, info] = trainNetwork(X_train,Y_train,layers,options);

%% Save network, training process and error
save(append(saveName,'.mat'),'net','info','maxEpochs','miniBatchSize','alpha0','tau','gamma','gradientThreshold')
