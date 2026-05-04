function mobilenet_2class_inference(inputFolder, outputFolder, trainedNetworkFile, labelFile)
% MOBILENET_2CLASS_INFERENCE
%
% Perform inference on Hartmann-Shack (HS) images using a pre-trained
% MobileNet-based binary classification model.
%
% The function processes all images in a given input folder, predicts
% their class (valid / invalid), and saves the results in organized
% output directories. Optionally, if a label file is provided, it
% computes classification metrics.
%
% Usage:
%   mobilenet_2class_inference(inputFolder, outputFolder, trainedNetworkFile)
%   mobilenet_2class_inference(inputFolder, outputFolder, trainedNetworkFile, labelFile)
%
% Inputs:
%   inputFolder         - Path to folder containing input images
%   outputFolder        - Path to folder where results will be saved
%   trainedNetworkFile  - MAT file containing trained network (variable: net)
%   labelFile           - (Optional) Excel file with columns:
%                         'filename' and 'label'
%
% Output:
%   - Images saved in:
%         outputFolder/valid/
%         outputFolder/invalid/
%   - Excel file with prediction results
%
% Notes:
%   - Binary classification: valid vs invalid
%   - Class names must match training labels: "invalid", "valid"
%   - Ground-truth labels can be numeric (0/1) or text ("valid"/"invalid")

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

    %% -------------------- Output folders --------------------
    validFolder = fullfile(outputFolder, 'valid');
    invalidFolder = fullfile(outputFolder, 'invalid');

    createFolderIfNeeded(validFolder);
    createFolderIfNeeded(invalidFolder);

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    excelFileName = fullfile(outputFolder, ...
        ['prediction_results_' timestamp '.xlsx']);

    %% -------------------- Load trained network --------------------
    networkData = load(trainedNetworkFile);

    if ~isfield(networkData, 'net')
        error('The trained model file must contain a variable named "net".');
    end

    net = networkData.net;

    % Class names must match the labels used during training
    classNames = ["invalid", "valid"];

    if ~isprop(net.Layers(1), 'InputSize')
        error('The loaded network does not contain a valid image input layer.');
    end

    inputSize = net.Layers(1).InputSize;

    %% -------------------- Load images --------------------
    imds = imageDatastore(inputFolder, ...
        'IncludeSubfolders', true, ...
        'FileExtensions', {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}, ...
        'ReadFcn', @readAndConvert);

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

        groundTruth = zeros(numImages, 1);
        predictions = zeros(numImages, 1);
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
            'PredictedLabel', NaN, ...
            'ConfidenceScore', NaN, ...
            'ElapsedTimeSec', NaN, ...
            'TrueLabel', NaN);
    else
        results(numImages, 1) = struct( ...
            'ImageName', "", ...
            'PredictedClassName', "", ...
            'PredictedLabel', NaN, ...
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

        % Resize image to match the input size required by network (e.g., 224×224 for MobileNet)
        imageForNetwork = imresize(imageForNetwork, inputSize(1:2));

        % Predict class scores
        scores = predict(net, imageForNetwork);
        scores = squeeze(scores);

        if isrow(scores)
            [maxScore, maxIdx] = max(scores);
        else
            [maxScore, maxIdx] = max(scores(:));
        end

        % Convert predicted index to class name
        if maxIdx <= numel(classNames)
            predictedClassName = classNames(maxIdx);
        else
            predictedClassName = "unknown";
        end

        % Convert class name to numeric label
        predictedNumeric = classNameToNumeric(predictedClassName);

        fprintf('  Predicted class: %s\n', predictedClassName);
        fprintf('  Predicted label: %d\n', predictedNumeric);
        fprintf('  Confidence score: %.4f\n', maxScore);

        % Read original image for saving
        originalImage = imread(currentFilePath);

        % Save into class-specific folder
        if predictedNumeric == 1
            imwrite(originalImage, fullfile(validFolder, fileName));
        else
            imwrite(originalImage, fullfile(invalidFolder, fileName));
        end

        % Optional ground-truth lookup
        trueLabelNumeric = NaN;
        if useGroundTruth
            trueLabelNumeric = getGroundTruthLabel(fileName, groundTruthTable);
            groundTruth(i) = trueLabelNumeric;
            predictions(i) = predictedNumeric;

            fprintf('  Ground truth label: %d\n', trueLabelNumeric);
        end

        % Store results
        results(i).ImageName = string(fileName);
        results(i).PredictedClassName = string(predictedClassName);
        results(i).PredictedLabel = predictedNumeric;
        results(i).ConfidenceScore = maxScore;
        results(i).ElapsedTimeSec = toc(imageTimer);

        if useGroundTruth
            results(i).TrueLabel = trueLabelNumeric;
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
        evaluateClassification(groundTruth, predictions);
    else
        fprintf('\nNo label file was provided. Confusion matrix and metrics were skipped.\n');
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

function imageOut = readAndConvert(filename)
% READANDCONVERT Read image and convert it to network-compatible format.
%
% This function:
%   1. Convert grayscale to 3-channel RGB (required by MobileNet)
%   2. Remove extra channels if present
%   3. Normalize pixel values to [0,1]

    imageOut = imread(filename);

    if ndims(imageOut) == 2
        imageOut = repmat(imageOut, [1 1 3]);
    elseif ndims(imageOut) == 3 && size(imageOut, 3) > 3
        imageOut = imageOut(:, :, 1:3);
    end

    imageOut = single(imageOut) / 255;
end

function numericLabel = classNameToNumeric(className)
% CLASSNAMETONUMERIC Convert class name to numeric label.
%
% Mapping:
%   valid   -> 1
%   invalid -> 0

    className = lower(string(className));

    if className == "valid"
        numericLabel = 1;
    elseif className == "invalid"
        numericLabel = 0;
    else
        numericLabel = 0;
    end
end

function trueLabel = getGroundTruthLabel(fileName, groundTruthTable)
% GETGROUNDTRUTHLABEL Retrieve ground-truth label (supports text or numeric)

    idx = find(strcmp(groundTruthTable.filename, fileName), 1);

    if isempty(idx)
        trueLabel = 0;
        fprintf('  Warning: true label not found for %s. Defaulting to invalid (0).\n', fileName);
        return;
    end

    labelValue = groundTruthTable.label(idx);

    % Case 1: numeric labels (0/1)
    if isnumeric(labelValue)
        trueLabel = labelValue;
        return;
    end

    % Case 2: text labels ("valid"/"invalid")
    labelStr = lower(string(labelValue));

    if labelStr == "valid"
        trueLabel = 1;
    elseif labelStr == "invalid"
        trueLabel = 0;
    else
        warning('Unknown label "%s" for %s. Defaulting to 0.', labelStr, fileName);
        trueLabel = 0;
    end
end

function evaluateClassification(groundTruth, predictions)
% EVALUATECLASSIFICATION Compute binary classification performance metrics.
%
% Convention:
%   1 = valid
%   0 = invalid

    if isempty(groundTruth) || isempty(predictions)
        fprintf('\nNot enough data to generate confusion matrix.\n');
        return;
    end

    if numel(groundTruth) ~= numel(predictions)
        error('Length mismatch: groundTruth and predictions must have the same length.');
    end

    C = confusionmat(double(groundTruth), double(predictions), 'Order', [0, 1]);

    % With Order = [0,1]:
    % C(1,1) = True invalid predicted invalid
    % C(1,2) = invalid predicted valid
    % C(2,1) = valid predicted invalid
    % C(2,2) = True valid predicted valid
    TN = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    TP = C(2,2);

    accuracy = (TP + TN) / max(1, sum(C(:)));
    sensitivity = TP / max(1, TP + FN);
    specificity = TN / max(1, TN + FP);
    precision = TP / max(1, TP + FP);

    if (precision + sensitivity) == 0
        f1Score = 0;
    else
        f1Score = 2 * (precision * sensitivity) / (precision + sensitivity);
    end

    fprintf('\n--- Confusion Matrix ---\n');
    disp(C);
    fprintf('Accuracy    : %.4f\n', accuracy);
    fprintf('Sensitivity : %.4f\n', sensitivity);
    fprintf('Specificity : %.4f\n', specificity);
    fprintf('Precision   : %.4f\n', precision);
    fprintf('F1 Score    : %.4f\n', f1Score);
end