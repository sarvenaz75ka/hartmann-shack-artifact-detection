function squeezenet_4class_inference(inputFolder, outputFolder, trainedNetworkFile, labelFile)
% SQUEEZENET_4CLASS_INFERENCE
%
% Perform inference on Hartmann-Shack (HS) images using a pre-trained
% SqueezeNet-based four-class classification model.
%
% The function processes all images in a given input folder, predicts
% their class, and saves the results in organized output directories.
% Optionally, if a label file is provided, it computes multiclass
% classification metrics.
%
% Usage:
%   squeezenet_4class_inference(inputFolder, outputFolder, trainedNetworkFile)
%   squeezenet_4class_inference(inputFolder, outputFolder, trainedNetworkFile, labelFile)
%
% Inputs:
%   inputFolder         - Path to folder containing input images
%   outputFolder        - Path to folder where results will be saved
%   trainedNetworkFile  - MAT file containing trained network (variable: net)
%   labelFile           - Optional Excel file with columns:
%                         'filename' and 'label'
%
% Output:
%   - Images saved in:
%         outputFolder/blinking/
%         outputFolder/corneal/
%         outputFolder/laser reflex/
%         outputFolder/valid/
%   - Excel file with prediction results
%
% Notes:
%   - Supported image formats: .bmp, .png, .jpg, .jpeg, .tif, .tiff
%   - This script is designed for inference only
%   - Ground-truth labels in Excel can be numeric IDs or text labels
%   - Class names must exactly match the labels used during training
%   - The class "laser reflex" in the dataset corresponds to "lens reflection"
%     in the manuscript. Both terms refer to the same artifact type.


    overallTimer = tic;

    %% -------------------- Input handling --------------------
    if nargin < 3
        error('You must provide inputFolder, outputFolder, and trainedNetworkFile.');
    end

    if nargin < 4
        labelFile = '';
    end

    if ~isfolder(inputFolder)
        error('Input folder does not exist: %s', inputFolder);
    end

    createFolderIfNeeded(outputFolder);

    if ~isfile(trainedNetworkFile)
        error('Trained network file does not exist: %s', trainedNetworkFile);
    end

    useGroundTruth = ~isempty(labelFile);
    if useGroundTruth && ~isfile(labelFile)
        error('Label file does not exist: %s', labelFile);
    end

    %% -------------------- Load trained network --------------------
    networkData = load(trainedNetworkFile);

    if ~isfield(networkData, 'net')
        error('The trained model file must contain a variable named "net".');
    end

    net = networkData.net;

    % Class names must exactly match the labels used during training
    classNames = ["blinking", "corneal", "laser reflex", "valid"];

    if ~isprop(net.Layers(1), 'InputSize')
        error('The loaded network does not contain a valid image input layer.');
    end

    inputSize = net.Layers(1).InputSize;
    numClasses = numel(classNames);

    %% -------------------- Output folders --------------------
    for c = 1:numClasses
        createFolderIfNeeded(fullfile(outputFolder, char(classNames(c))));
    end

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    excelFileName = fullfile(outputFolder, ...
        ['prediction_results_' timestamp '.xlsx']);

    %% -------------------- Load images --------------------
    imds = imageDatastore(inputFolder, ...
        'IncludeSubfolders', true, ...
        'FileExtensions', {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}, ...
        'ReadFcn', @readAndConvertForSqueezeNet);

    numImages = numel(imds.Files);

    if numImages == 0
        error('No supported image files were found in: %s', inputFolder);
    end

    fprintf('Found %d images in input folder.\n', numImages);

    %% -------------------- Optional label loading --------------------
    if useGroundTruth
        groundTruthTable = readtable(labelFile);

        requiredVars = {'filename', 'label'};
        for k = 1:numel(requiredVars)
            if ~ismember(requiredVars{k}, groundTruthTable.Properties.VariableNames)
                error('Label file must contain columns named "filename" and "label".');
            end
        end

        groundTruth = NaN(numImages, 1);
        predictions = NaN(numImages, 1);
    else
        groundTruthTable = table();
        groundTruth = [];
        predictions = [];
    end

    %% -------------------- Prepare results --------------------
    if useGroundTruth
        results(numImages, 1) = struct( ...
            'ImageName', "", ...
            'PredictedClassName', "", ...
            'PredictedLabelID', NaN, ...
            'ConfidenceScore', NaN, ...
            'ElapsedTimeSec', NaN, ...
            'TrueLabelID', NaN);
    else
        results(numImages, 1) = struct( ...
            'ImageName', "", ...
            'PredictedClassName', "", ...
            'PredictedLabelID', NaN, ...
            'ConfidenceScore', NaN, ...
            'ElapsedTimeSec', NaN);
    end

    %% -------------------- Main inference loop --------------------
    for i = 1:numImages
        imageTimer = tic;

        currentFilePath = imds.Files{i};
        [~, baseName, ext] = fileparts(currentFilePath);
        fileName = [baseName ext];

        fprintf('\nProcessing image %d/%d: %s\n', i, numImages, fileName);

        % Read and preprocess image for network input
        imageForNetwork = readimage(imds, i);

        % Resize image to match the input size required by SqueezeNet
        % (typically 227x227x3)
        imageForNetwork = imresize(imageForNetwork, inputSize(1:2));

        % Predict class scores
        scores = predict(net, imageForNetwork);
        scores = squeeze(scores);

        if isrow(scores)
            [maxScore, maxIdx] = max(scores);
        else
            [maxScore, maxIdx] = max(scores(:));
        end

        % Convert predicted index to class name / numeric ID
        if maxIdx <= numClasses
            predictedClassName = classNames(maxIdx);
            predictedLabelID = maxIdx;
        else
            predictedClassName = "unknown";
            predictedLabelID = NaN;
        end

        fprintf('  Predicted class: %s\n', predictedClassName);
        fprintf('  Predicted label ID: %d\n', predictedLabelID);
        fprintf('  Confidence score: %.4f\n', maxScore);

        % Read original image for saving
        originalImage = imread(currentFilePath);

        % Save into class-specific folder
        if predictedClassName ~= "unknown"
            outputClassFolder = fullfile(outputFolder, char(predictedClassName));
        else
            outputClassFolder = fullfile(outputFolder, 'unknown');
            createFolderIfNeeded(outputClassFolder);
        end
        imwrite(originalImage, fullfile(outputClassFolder, fileName));

        % Optional ground-truth lookup
        trueLabelID = NaN;
        if useGroundTruth
            trueLabelID = getGroundTruthLabel4Class(fileName, groundTruthTable, classNames);
            groundTruth(i) = trueLabelID;
            predictions(i) = predictedLabelID;

            if ~isnan(trueLabelID)
                fprintf('  Ground truth label ID: %d\n', trueLabelID);
            else
                fprintf('  Ground truth label ID: NaN\n');
            end
        end

        % Store results
        results(i).ImageName = string(fileName);
        results(i).PredictedClassName = string(predictedClassName);
        results(i).PredictedLabelID = predictedLabelID;
        results(i).ConfidenceScore = maxScore;
        results(i).ElapsedTimeSec = toc(imageTimer);

        if useGroundTruth
            results(i).TrueLabelID = trueLabelID;
        end
    end

    %% -------------------- Save results --------------------
    resultsTable = struct2table(results);

    try
        writetable(resultsTable, excelFileName);
        fprintf('\nProcessing complete.\n');
        fprintf('Results saved to: %s\n', excelFileName);

    catch ME
        warning('Could not write results to the default Excel file.');
        fprintf('Reason: %s\n', ME.message);

        try
            backupTimestamp = datestr(now, 'yyyymmdd_HHMMSS');
            fallbackFile = fullfile(outputFolder, ...
                ['prediction_results_backup_' backupTimestamp '.xlsx']);

            writetable(resultsTable, fallbackFile);
            fprintf('Results were saved to backup file:\n%s\n', fallbackFile);

        catch ME2
            warning('Could not write results to the backup Excel file either.');
            fprintf('Backup reason: %s\n', ME2.message);
        end
    end

    %% -------------------- Optional evaluation --------------------
    if useGroundTruth
        evaluateMulticlassClassification(groundTruth, predictions, classNames);
    else
        fprintf('\nNo label file was provided. Multiclass evaluation was skipped.\n');
    end

    fprintf('\nTotal elapsed time: %.2f seconds\n', toc(overallTimer));
end

%% =====================================================================
%% Helper functions
%% =====================================================================

function createFolderIfNeeded(folderPath)
% CREATEFOLDERIFNEEDED Create folder if it does not already exist.

    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
end

function imageOut = readAndConvertForSqueezeNet(filename)
% READANDCONVERTFORSQUEEZENET Read image and apply SqueezeNet preprocessing.
%
% This function:
%   1. Reads the input image from disk
%   2. Converts grayscale images to 3-channel RGB
%   3. Converts data type to single
%   4. Applies ImageNet channel-wise normalization


    imageOut = imread(filename);

    % Convert grayscale images to 3 channels
    if ndims(imageOut) == 2
        imageOut = repmat(imageOut, [1 1 3]);
    elseif ndims(imageOut) == 3 && size(imageOut, 3) > 3
        imageOut = imageOut(:, :, 1:3);
    end

    % Convert to single precision
    imageOut = single(imageOut);

    % ImageNet mean and standard deviation (scaled to [0,255] domain)
    mu = [0.485, 0.456, 0.406] * 255;
    sigma = [0.229, 0.224, 0.225] * 255;

    % Apply channel-wise normalization
    for channelIdx = 1:3
        imageOut(:, :, channelIdx) = ...
            (imageOut(:, :, channelIdx) - mu(channelIdx)) / sigma(channelIdx);
    end
end

function trueLabelID = getGroundTruthLabel4Class(fileName, groundTruthTable, classNames)
% GETGROUNDTRUTHLABEL4CLASS Retrieve multiclass ground-truth label.
%
% Supports:
%   - numeric IDs: 1..N
%   - text labels matching classNames

    idx = find(strcmp(groundTruthTable.filename, fileName), 1);

    if isempty(idx)
        trueLabelID = NaN;
        fprintf('  Warning: true label not found for %s.\n', fileName);
        return;
    end

    labelValue = groundTruthTable.label(idx);

    if isnumeric(labelValue)
        trueLabelID = labelValue;
        return;
    end

    labelStr = lower(strtrim(string(labelValue)));
    classNamesLower = lower(strtrim(classNames));

    matchIdx = find(classNamesLower == labelStr, 1);

    if isempty(matchIdx)
        warning('Unknown label "%s" for %s.', labelStr, fileName);
        trueLabelID = NaN;
    else
        trueLabelID = matchIdx;
    end
end

function evaluateMulticlassClassification(groundTruth, predictions, classNames)
% EVALUATEMULTICLASSCLASSIFICATION Compute multiclass performance metrics.
%
% Reports:
%   - Confusion matrix
%   - Overall accuracy
%   - Per-class precision, recall, and F1 score
%   - Macro-averaged precision, recall, and F1 score

    validIdx = ~isnan(groundTruth) & ~isnan(predictions);
    groundTruth = groundTruth(validIdx);
    predictions = predictions(validIdx);

    if isempty(groundTruth) || isempty(predictions)
        fprintf('\nNot enough valid data to generate multiclass metrics.\n');
        return;
    end

    if numel(groundTruth) ~= numel(predictions)
        error('Length mismatch: groundTruth and predictions must have the same length.');
    end

    numClasses = numel(classNames);
    labelOrder = 1:numClasses;

    C = confusionmat(double(groundTruth), double(predictions), 'Order', labelOrder);

    accuracy = sum(diag(C)) / max(1, sum(C(:)));

    precisionPerClass = zeros(numClasses, 1);
    recallPerClass = zeros(numClasses, 1);
    f1PerClass = zeros(numClasses, 1);

    for c = 1:numClasses
        TP = C(c, c);
        FP = sum(C(:, c)) - TP;
        FN = sum(C(c, :)) - TP;

        precisionPerClass(c) = TP / max(1, TP + FP);
        recallPerClass(c) = TP / max(1, TP + FN);

        if precisionPerClass(c) + recallPerClass(c) == 0
            f1PerClass(c) = 0;
        else
            f1PerClass(c) = 2 * (precisionPerClass(c) * recallPerClass(c)) / ...
                (precisionPerClass(c) + recallPerClass(c));
        end
    end

    macroPrecision = mean(precisionPerClass);
    macroRecall = mean(recallPerClass);
    macroF1 = mean(f1PerClass);

    fprintf('\n--- Confusion Matrix ---\n');
    disp(C);

    fprintf('Overall Accuracy : %.4f\n', accuracy);
    fprintf('Macro Precision  : %.4f\n', macroPrecision);
    fprintf('Macro Recall     : %.4f\n', macroRecall);
    fprintf('Macro F1 Score   : %.4f\n', macroF1);

    fprintf('\n--- Per-class Metrics ---\n');
    for c = 1:numClasses
        fprintf('%s -> Precision: %.4f | Recall: %.4f | F1: %.4f\n', ...
            classNames(c), precisionPerClass(c), recallPerClass(c), f1PerClass(c));
    end
end