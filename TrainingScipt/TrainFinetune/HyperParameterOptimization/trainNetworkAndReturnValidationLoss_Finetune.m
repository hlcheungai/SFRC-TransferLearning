function FinalValidationLoss = trainNetworkAndReturnValidationLoss_Finetune(params,dataset,net,folderPath,baseFileName,display)
%% Import hyper-parameters to try
maxEpochs = params.maxEpochs;                   % Maximum number of training epochs.
miniBatchSize = params.miniBatchSize;           % Mini batch size.
alpha0 = params.alpha0;                         % InitialLearnRate
tau = params.tau;                               % LearnRateDropPeriod
gamma = params.gamma;                           % LearnRateDropFactor
gradientThreshold = params.gradientThreshold;   % GradientThreshold

%% Import data
X_train = dataset.X_train;
Y_train = dataset.Y_train;
X_valid = dataset.X_valid;
Y_valid = dataset.Y_valid;

%% Define training options
VBfreq = floor(size(X_train,1)/miniBatchSize);
options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'Shuffle','every-epoch', ...
    'ResetInputNormalization', false,...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',gradientThreshold, ...
    'InitialLearnRate',alpha0, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',tau, ...
    'LearnRateDropFactor',gamma, ...
    'Verbose',true, ...
    'VerboseFrequency',VBfreq, ...
    'Plots',display, ...
    'ValidationFrequency',VBfreq, ...
    'ValidationData',{X_valid,Y_valid}, ...
    'OutputNetwork','best-validation-loss');

layers = [
    sequenceInputLayer(13,"Name","sequence","Normalization","zscore")
    gruLayer(500,'Name','gru_1')
    gruLayer(500,'Name','gru_2')
    gruLayer(500,'Name','gru_3')
    dropoutLayer(0.5,'Name','dropout')
    fullyConnectedLayer(6,'Name','fc')
    regressionLayer("Name","regressionoutput")];
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
%% Train network and get final validation loss
clear net
[net,info] = trainNetwork(X_train,Y_train,layers,options);
FinalValidationLoss = info.FinalValidationLoss;

%% Save file
index = 1;
while true
    fileName = sprintf('%s_%d.mat', baseFileName, index);
    if exist(fullfile(folderPath, fileName), 'file')
        index = index + 1;
    else
        break;
    end
end
save(fullfile(folderPath, fileName), "net","info","params","dataset","network");
