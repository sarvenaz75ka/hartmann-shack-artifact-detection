function correlation_artifact_detection(inputFolder, outputFolder, meanImagePath, correlationThreshold, labelFile)
% RUN_CORRELATION_ARTIFACT_DETECTION
%
% Classify Hartmann-Shack (HS) images into valid and invalid categories
% using a correlation-based approach combined with pupil centering and
% reflection detection.
%
% Processing pipeline:
%   1. Load all images from the input folder
%   2. Detect the pupil using morphological operations
%   3. Estimate pupil centroid and diameter
%   4. Translate the image to center the detected pupil
%   5. Compute the correlation with a reference mean image
%   6. Detect possible bright reflections
%   7. Classify each image as Valid or Invalid
%   8. Save classified images into output subfolders
%   9. Export image-wise processing results to an Excel file
%  10. Optionally, if a label file is provided, compute classification
%      metrics and a confusion matrix
%
% Usage:
%   correlation_artifact_detection(inputFolder, outputFolder, meanImagePath)
%   correlation_artifact_detection(inputFolder, outputFolder, meanImagePath, correlationThreshold)
%   correlation_artifact_detection(inputFolder, outputFolder, meanImagePath, correlationThreshold, labelFile)
%
% Inputs:
%   inputFolder           - Path to the folder containing input images
%   outputFolder          - Path to the folder where results will be saved
%   meanImagePath         - Path to the reference mean image
%   correlationThreshold  - Optional threshold for correlation (default: 0.70)
%   labelFile             - Optional Excel file with columns:
%                           'filename' and 'label'
%
% Output structure:
%   outputFolder/
%       ├── Valid/
%       ├── Invalid/
%       └── correlation_results_yyyymmdd_HHMMSS.xlsx
%
% Notes:
%   - Supported image formats: .bmp, .png, .jpg, .jpeg, .tif, .tiff
%   - Ground-truth evaluation is optional
%   - Labels are assumed to follow:
%       1 = Valid
%       0 = Invalid

    overallTimer = tic;

    %% -------------------- Input handling --------------------
    if nargin < 3
        error('You must provide inputFolder, outputFolder, and meanImagePath.');
    end

        % Default correlation threshold:
    % If not provided by the user, a value of 0.70 is used.
    % This value was empirically selected to provide a balance between
    % accepting valid images and rejecting artifact-corrupted patterns.
    if nargin < 4 || isempty(correlationThreshold)
        correlationThreshold = 0.70;
    end

    if nargin < 5
        labelFile = '';
    end

    if ~isfolder(inputFolder)
        error('Input folder does not exist: %s', inputFolder);
    end

    if ~isfile(meanImagePath)
        error('Mean image file does not exist: %s', meanImagePath);
    end

    useGroundTruth = ~isempty(labelFile);
    if useGroundTruth && ~isfile(labelFile)
        error('Label file does not exist: %s', labelFile);
    end

    %% -------------------- Parameters --------------------
    params.pixelToMM = 0.008833;
    params.centerTolerancePx = 10;
    params.pupilOpenSE = strel('disk', 25);

    %% -------------------- Output folders --------------------
    validFolder = fullfile(outputFolder, 'Valid');
    invalidFolder = fullfile(outputFolder, 'Invalid');

    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    excelFileName = fullfile(outputFolder, ...
        ['correlation_results_' timestamp '.xlsx']);

    createFolderIfNeeded(outputFolder);
    createFolderIfNeeded(validFolder);
    createFolderIfNeeded(invalidFolder);

    %% -------------------- Load images --------------------
    imds = imageDatastore(inputFolder, ...
        'IncludeSubfolders', true, ...
        'FileExtensions', {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'});

    numImages = numel(imds.Files);
    if numImages == 0
        error('No supported image files were found in: %s', inputFolder);
    end

    fprintf('Found %d images in input folder.\n', numImages);

    %% -------------------- Load reference mean image --------------------
    meanImage = imread(meanImagePath);
    if ndims(meanImage) == 3
        meanImage = rgb2gray(meanImage(:,:,1:3));
    end
    meanImage = double(meanImage);

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
            'PupilDiameterMM', NaN, ...
            'WasCentered', false, ...
            'CorrelationCoefficient', NaN, ...
            'HasReflection', false, ...
            'PredictedStatus', "", ...
            'ElapsedTimeSec', NaN, ...
            'TrueStatus', "");
    else
        results(numImages, 1) = struct( ...
            'ImageName', "", ...
            'PupilDiameterMM', NaN, ...
            'WasCentered', false, ...
            'CorrelationCoefficient', NaN, ...
            'HasReflection', false, ...
            'PredictedStatus', "", ...
            'ElapsedTimeSec', NaN);
    end

    %% -------------------- Main processing loop --------------------
    for i = 1:numImages
        imageTimer = tic;

        imagePath = imds.Files{i};
        [~, baseName, ext] = fileparts(imagePath);
        fileName = [baseName ext];

        fprintf('\nProcessing image %d/%d: %s\n', i, numImages, fileName);

        image = readimage(imds, i);
        if ndims(image) == 3
            image = rgb2gray(image(:,:,1:3));
        end

        predictedStatus = "Invalid";
        predictedNumeric = 0;
        estimatedPupilDiameterMM = NaN;
        correlationCoefficient = NaN;
        wasCentered = false;
        hasReflection = false;
        trueStatusString = "";

        % Detect pupil and estimate its geometry
        pupilData = detectPupilForCorrelation(image, params);

        currentImageForCorrelation = double(image);

        if pupilData.found
            estimatedPupilDiameterMM = pupilData.diameterMM;
            fprintf('  Estimated pupil diameter: %.2f mm\n', estimatedPupilDiameterMM);

            % Center image based on detected pupil centroid
            [currentImageForCorrelation, wasCentered] = centerImageOnPupil( ...
                image, pupilData.centroid, params.centerTolerancePx);

            if wasCentered
                fprintf('  Centering: PASS\n');
            else
                fprintf('  Centering: FAIL\n');
            end
        else
            fprintf('  No valid pupil detected.\n');
        end

        % Proceed only if pupil was detected and image could be centered
        if pupilData.found && wasCentered

            % Check that reference image and current image have matching dimensions
            if isequal(size(meanImage), size(currentImageForCorrelation))
                correlationCoefficient = corr(meanImage(:), currentImageForCorrelation(:));
                fprintf('  Correlation coefficient: %.4f\n', correlationCoefficient);

                if correlationCoefficient >= correlationThreshold
                    hasReflection = detectReflection(image);

                    if hasReflection
                        predictedStatus = "Invalid (Reflection Detected)";
                        predictedNumeric = 0;
                        fprintf('  Reflection detected: image classified as Invalid.\n');
                    else
                        predictedStatus = "Valid";
                        predictedNumeric = 1;
                        fprintf('  Passed correlation threshold and no reflection detected.\n');
                    end
                else
                    predictedStatus = "Invalid (Low Correlation)";
                    predictedNumeric = 0;
                    fprintf('  Correlation below threshold: image classified as Invalid.\n');
                end
            else
                predictedStatus = "Invalid (Dimension Mismatch)";
                predictedNumeric = 0;
                fprintf('  ERROR: Image dimensions do not match mean image.\n');
            end
        else
            predictedStatus = "Invalid (Pupil Not Detected/Centered)";
            predictedNumeric = 0;
            fprintf('  Image classified as Invalid because pupil was not reliably detected/centered.\n');
        end

        % Save original image into predicted class folder
        if predictedNumeric == 1
            imwrite(image, fullfile(validFolder, fileName));
        else
            imwrite(image, fullfile(invalidFolder, fileName));
        end

        fprintf('  Predicted: %s\n', predictedStatus);

        % Optional ground-truth lookup
        if useGroundTruth
            [groundTruth(i), trueStatusString] = getGroundTruthLabel(fileName, groundTruthTable);
            predictions(i) = predictedNumeric;
            fprintf('  Ground truth: %s (%d)\n', trueStatusString, groundTruth(i));
        end

        % Store results
        results(i).ImageName = string(fileName);
        results(i).PupilDiameterMM = estimatedPupilDiameterMM;
        results(i).WasCentered = wasCentered;
        results(i).CorrelationCoefficient = correlationCoefficient;
        results(i).HasReflection = hasReflection;
        results(i).PredictedStatus = predictedStatus;
        results(i).ElapsedTimeSec = toc(imageTimer);

        if useGroundTruth
            results(i).TrueStatus = string(trueStatusString);
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
                ['correlation_results_backup_' backupTimestamp '.xlsx']);

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
% CREATEFOLDERIFNEEDED Create a folder if it does not already exist.

    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
end

function pupilData = detectPupilForCorrelation(image, params)
% DETECTPUPILFORCORRELATION Detect pupil candidate for correlation pipeline.
%
% This function uses morphological opening, binarization, hole filling,
% and connected-component analysis to estimate the pupil centroid and
% equivalent diameter.

    pupilData = struct( ...
        'found', false, ...
        'centroid', [NaN, NaN], ...
        'diameterMM', NaN);

    processedImage = imopen(image, params.pupilOpenSE);
    binaryImage = imbinarize(processedImage);
    filledImage = imfill(binaryImage, 'holes');

    cc = bwconncomp(filledImage);
    if cc.NumObjects == 0
        return;
    end

    numPixels = cellfun(@numel, cc.PixelIdxList);
    [largestSize, idxLargest] = max(numPixels);

    if isempty(largestSize) || largestSize <= 0
        return;
    end

    largestComponent = false(size(filledImage));
    largestComponent(cc.PixelIdxList{idxLargest}) = true;

    props = regionprops(largestComponent, 'Centroid', 'Area', 'EquivDiameter');
    if isempty(props) || props(1).Area <= 0
        return;
    end

    pupilData.found = true;
    pupilData.centroid = props(1).Centroid;
    pupilData.diameterMM = props(1).EquivDiameter * params.pixelToMM;
end

function [centeredImage, wasCentered] = centerImageOnPupil(image, pupilCentroid, tolerance)
% CENTERIMAGEONPUPIL Translate image so that the pupil centroid is near image center.
%
% Inputs:
%   image         - Input grayscale image
%   pupilCentroid - Detected pupil centroid [x, y]
%   tolerance     - Allowed offset from image center in pixels
%
% Outputs:
%   centeredImage - Centered image
%   wasCentered   - True if centering was successful or already satisfied

    centeredImage = double(image);
    wasCentered = false;

    if numel(pupilCentroid) ~= 2 || any(isnan(pupilCentroid))
        return;
    end

    [rows, cols] = size(image);
    imageCenterX = round(cols / 2);
    imageCenterY = round(rows / 2);

    centroidX = round(pupilCentroid(1));
    centroidY = round(pupilCentroid(2));

    if abs(centroidX - imageCenterX) <= tolerance && ...
       abs(centroidY - imageCenterY) <= tolerance

        centeredImage = double(image);
        wasCentered = true;
    else
        deltaX = imageCenterX - centroidX;
        deltaY = imageCenterY - centroidY;

        centeredImage = imtranslate(image, [deltaX, deltaY], 'FillValues', 0);
        centeredImage = double(centeredImage);
        wasCentered = true;
    end
end

function [trueLabel, trueStatusString] = getGroundTruthLabel(fileName, groundTruthTable)
% GETGROUNDTRUTHLABEL Retrieve ground-truth label for a given image file.

    idx = find(strcmp(groundTruthTable.filename, fileName), 1);

    if isempty(idx)
        trueLabel = 0;
        trueStatusString = "N/A (Not Found)";
        fprintf('  Warning: true label not found for %s. Defaulting to Invalid.\n', fileName);
        return;
    end

    trueLabel = groundTruthTable.label(idx);

    if trueLabel == 1
        trueStatusString = "Valid";
    elseif trueLabel == 0
        trueStatusString = "Invalid";
    else
        trueStatusString = "Unknown";
    end
end

function evaluateClassification(groundTruth, predictions)
% EVALUATECLASSIFICATION Compute binary classification performance metrics.
%
% Convention:
%   1 = Valid
%   0 = Invalid

    if isempty(groundTruth) || isempty(predictions)
        fprintf('\nNot enough data to generate confusion matrix.\n');
        return;
    end

    if numel(groundTruth) ~= numel(predictions)
        error('Length mismatch: groundTruth and predictions must have the same length.');
    end

    C = confusionmat(double(groundTruth), double(predictions), 'Order', [0, 1]);

    % With Order = [0,1]:
    % C(1,1) = True Invalid predicted Invalid
    % C(1,2) = Invalid predicted Valid
    % C(2,1) = Valid predicted Invalid
    % C(2,2) = True Valid predicted Valid
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

function reflectionDetected = detectReflection(image)
% DETECTREFLECTION Detect bright circular reflections using imfindcircles.
%
% This function searches for bright circular structures that may correspond
% to reflection artifacts.

    if ndims(image) == 3
        grayImage = rgb2gray(image(:,:,1:3));
    else
        grayImage = image;
    end

    if ~isa(grayImage, 'uint8')
        grayImage = im2uint8(grayImage);
    end

    [centers, ~, ~] = imfindcircles(grayImage, [20 90], ...
        'ObjectPolarity', 'bright', ...
        'Sensitivity', 0.9);

    reflectionDetected = ~isempty(centers);
end