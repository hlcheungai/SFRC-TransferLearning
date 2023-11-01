%% This is the parameter obtain from hyperparameter tuning (Finetune)
% Define training parameters
maxEpochs = 343;                                % Maximum number of training epochs.
miniBatchSize = 40;                             % Mini batch size.
alpha0 = 0.0003;                                % InitialLearnRate
tau = 23;                                      % LearnRateDropPeriod
gamma = 0.9735;                                    % LearnRateDropFactor
VBfreq = floor(size(X_train,1)/miniBatchSize);  % ValidationFrequency and VerboseFrequency
resetInputNormalization = false;                % ResetInputNormalization
gradientThreshold = 1.0538;                          % GradientThreshold
shuffle = 'every-epoch';
outputNetwork = 'best-validation-loss';
