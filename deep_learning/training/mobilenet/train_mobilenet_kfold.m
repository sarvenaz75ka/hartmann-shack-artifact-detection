function train_mobilenet_kfold(dataFolder, foldFolder, baseNetworkFile, outputFolder, numFolds)
% TRAIN_MOBILENET_KFOLD
%
% Train and evaluate a MobileNet-based using k-fold
% cross-validation.
%
% Usage:
%   train_mobilenet_kfold(dataFolder, foldFolder, baseNetworkFile, outputFolder)
%   train_mobilenet_kfold(dataFolder, foldFolder, baseNetworkFile, outputFolder, numFolds)
%
% Inputs:
%   dataFolder       - Root folder containing the full labeled dataset
%   foldFolder       - Folder containing Fold_1_Data.mat, Fold_2_Data.mat, ...
%   baseNetworkFile  - MAT file containing the base network variable: net
%   outputFolder     - Folder where results and trained models will be saved
%   numFolds         - Number of folds (default: 5)

    overallTimer = tic;

    %% -------------------- Input handling --------------------
    if nargin < 4
        error('You must provide dataFolder, foldFolder, baseNetworkFile, and outputFolder.');
    end

    if nargin < 5 || isempty(numFolds)
        numFolds = 5;
    end

    if ~isfolder(dataFolder)
        error('Data folder does not exist: %s', dataFolder);
    end

    if ~isfolder(foldFolder)
        error('Fold folder does not exist: %s', foldFolder);
    end

    if ~isfile(baseNetworkFile)
        error('Base network file does not exist: %s', baseNetworkFile);
    end

    createFolderIfNeeded(outputFolder);
    modelsFolder = fullfile(outputFolder, 'trained_models');
    createFolderIfNeeded(modelsFolder);

    %% -------------------- Load dataset metadata --------------------
    imds = imageDatastore(dataFolder, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'ReadFcn', @readAndConvert);

    if isempty(imds.Files)
        error('No images found in data folder.');
    end

    classNames = categories(imds.Labels);
    numClasses = numel(classNames);

    fprintf('Found %d images and %d classes.\n', numel(imds.Files), numClasses);
    disp('Class names:');
    disp(classNames);

    %% -------------------- Load base network --------------------
    networkData = load(baseNetworkFile);

    if ~isfield(networkData, 'net')
        error('The base network file must contain a variable named "net".');
    end

    baseNet = networkData.net;

    if ~isprop(baseNet.Layers(1), 'InputSize')
        error('The loaded network does not contain a valid image input layer.');
    end

    inputSize = baseNet.Layers(1).InputSize;

    %% -------------------- Storage --------------------
    accuracyValues = zeros(numFolds, 1);
    foldResults(numFolds, 1) = struct( ...
        'Fold', NaN, ...
        'NumTrainImages', NaN, ...
        'NumTestImages', NaN, ...
        'Accuracy', NaN, ...
        'ElapsedTimeSec', NaN);

    %% -------------------- Cross-validation loop --------------------
    for foldIdx = 1:numFolds
        foldTimer = tic;

        foldFile = fullfile(foldFolder, sprintf('Fold_%d_Data.mat', foldIdx));
        if ~isfile(foldFile)
            error('Fold file not found: %s', foldFile);
        end

        foldData = load(foldFile);

        if ~isfield(foldData, 'imdsTrain') || ~isfield(foldData, 'imdsTest')
            error('Fold file %s must contain variables "imdsTrain" and "imdsTest".', foldFile);
        end

        imdsTrain = foldData.imdsTrain;
        imdsTest = foldData.imdsTest;

        imdsTrain.ReadFcn = @readAndConvert;
        imdsTest.ReadFcn = @readAndConvert;

        fprintf('\nStarting training and evaluation for Fold %d of %d...\n', foldIdx, numFolds);

        augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
        augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

        options = trainingOptions("adam", ...
            MaxEpochs = 12, ...
            Plots = "training-progress", ...
            Metrics = "accuracy", ...
            Verbose = false, ...
            ExecutionEnvironment = "gpu");

        % Important: start each fold from the same base network
        net = trainnet(augimdsTrain, baseNet, "crossentropy", options);

        % Save trained model for this fold
        modelFile = fullfile(modelsFolder, sprintf('mobilenet_4class_fold_%d.mat', foldIdx));
        save(modelFile, 'net');

        % Predict on test set
        YPred = minibatchpredict(net, augimdsTest);
        [~, predictedLabels] = max(YPred, [], 2);
        YPredCategorical = categorical(predictedLabels, 1:numClasses, classNames);

        YTest = imdsTest.Labels;

        % Show confusion matrix
        figure;
        confusionchart(YTest, YPredCategorical);
        title(sprintf('MobileNet - Fold %d', foldIdx));

        accuracy = mean(YPredCategorical == YTest);
        accuracyValues(foldIdx) = accuracy;

        foldResults(foldIdx).Fold = foldIdx;
        foldResults(foldIdx).NumTrainImages = numel(imdsTrain.Files);
        foldResults(foldIdx).NumTestImages = numel(imdsTest.Files);
        foldResults(foldIdx).Accuracy = accuracy;
        foldResults(foldIdx).ElapsedTimeSec = toc(foldTimer);

        fprintf('Completed Fold %d. Test Accuracy: %.2f%%\n', foldIdx, accuracy * 100);
    end

    %% -------------------- Final summary --------------------
    meanAccuracy = mean(accuracyValues);
    stdAccuracy = std(accuracyValues);

    fprintf('\n----------------------------------------\n');
    fprintf('Average %d-Fold Accuracy : %.2f%% (+/- %.2f%%)\n', ...
        numFolds, meanAccuracy * 100, stdAccuracy * 100);
    fprintf('----------------------------------------\n');

    resultsTable = struct2table(foldResults);
    summaryFile = fullfile(outputFolder, 'mobilenet_kfold_results.xlsx');
    writetable(resultsTable, summaryFile);

    fprintf('Results saved to: %s\n', summaryFile);
    fprintf('Total elapsed time: %.2f seconds\n', toc(overallTimer));
end

%% =====================================================================
%% Helper functions
%% =====================================================================

function createFolderIfNeeded(folderPath)
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
end

function I = readAndConvert(filename)
% READANDCONVERT Read image and convert it to MobileNet-compatible format.
%
% This function:
%   1. Reads the image
%   2. Converts grayscale images to 3 channels
%   3. Converts to single precision in [0,1]

    I = imread(filename);

    if ndims(I) == 2
        I = repmat(I, [1 1 3]);
    elseif ndims(I) == 3 && size(I, 3) > 3
        I = I(:, :, 1:3);
    end

    I = single(I) / 255;
end