function run_morphological_artifact_detection(inputFolder, outputFolder, labelFile)
% RUN_ARTIFACT_CLASSIFICATION
%
% Classify Hartmann-Shack (HS) images into valid and invalid categories
% based on morphology-based pupil detection and corneal reflection analysis.
%
% Processing pipeline:
%   1. Load all images from the input folder
%   2. Detect the pupil region using morphological operations
%   3. Evaluate pupil geometry based on circularity and diameter
%   4. Detect bright corneal reflections inside the pupil region
%   5. Classify each image as Valid or Invalid
%   6. Save classified images into output subfolders
%   7. Export image-wise processing results to an Excel file
%   8. Optionally, if a label file is provided, compute classification
%      metrics and a confusion matrix
%
% Usage:
%   run_morphological_artifact_detection(inputFolder, outputFolder)
%   run_morphological_artifact_detection(inputFolder, outputFolder, labelFile)
%
% Inputs:
%   inputFolder  - Path to the folder containing the input images
%   outputFolder - Path to the folder where results will be saved
%   labelFile    - Optional Excel file containing ground-truth labels
%
% Output structure:
%   outputFolder/
%       ├── Valid/
%       ├── Invalid/
%       ├── Reflection_Visualizations/
%       └── processing_results_yyyymmdd_HHMMSS.xlsx
%
% Notes:
%   - Supported image formats: .bmp, .png, .jpg, .jpeg, .tif, .tiff
%   - Ground-truth evaluation is optional
%   - Reflection overlays are only saved for images classified as invalid
%     due to detected corneal reflections

   
    overallTimer = tic;

    %% -------------------- Input handling --------------------
    if nargin < 2
        error('You must provide at least inputFolder and outputFolder.');
    end

    if nargin < 3
        labelFile = '';
    end

    if ~isfolder(inputFolder)
        error('Input folder does not exist: %s', inputFolder);
    end

    useGroundTruth = ~isempty(labelFile);

    if useGroundTruth && ~isfile(labelFile)
        error('Label file does not exist: %s', labelFile);
    end

    %% -------------------- Parameters --------------------
    % Conversion factor from pixels to millimeters
    params.pixelToMM = 0.008833;

    % Minimum acceptable pupil diameter
    params.minPupilDiameterMM = 3.0;

    % Minimum number of bright pixels required to confirm a reflection
    params.reflectionMinPixelCount = 1000;

    % Fraction of pupil radius used to define the reflection search region
    params.outerReflectionSearchFactor = 0.6;

    % Structuring element used for morphological opening
    params.reflectionOpenSE = strel('disk', 25);

    % Geometric validation thresholds
    params.eccentricityThreshold = 0.5;
    params.areaDifferenceThreshold = 0.11;

    %% -------------------- Output folders --------------------
    validFolder = fullfile(outputFolder, 'Valid');
    invalidFolder = fullfile(outputFolder, 'Invalid');
    reflectionVizFolder = fullfile(outputFolder, 'Reflection_Visualizations');

    % Use a timestamped Excel filename to avoid overwrite/lock conflicts
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    excelFileName = fullfile(outputFolder, ...
        ['processing_results_' timestamp '.xlsx']);

    createFolderIfNeeded(outputFolder);
    createFolderIfNeeded(validFolder);
    createFolderIfNeeded(invalidFolder);
    createFolderIfNeeded(reflectionVizFolder);

    %% -------------------- Load images --------------------
    % Load all supported images from the input folder and its subfolders
    imds = imageDatastore(inputFolder, ...
        'IncludeSubfolders', true, ...
        'FileExtensions', {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'});

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
            'PupilDiameterMM', NaN, ...
            'IsCircle', false, ...
            'PredictedStatus', "", ...
            'HasReflection', false, ...
            'ElapsedTimeSec', NaN, ...
            'TrueStatus', "");
    else
        results(numImages, 1) = struct( ...
            'ImageName', "", ...
            'PupilDiameterMM', NaN, ...
            'IsCircle', false, ...
            'PredictedStatus', "", ...
            'HasReflection', false, ...
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
        if size(image, 3) == 3
            image = rgb2gray(image);
        end

        predictedStatus = "Invalid";
        estimatedPupilDiameterMM = NaN;
        isCircle = false;
        hasReflection = false;
        trueStatusString = "";

        % Estimate pupil location and geometry from the current image
        pupilData = detectPupilRegion(image, params);

        if pupilData.found
            estimatedPupilDiameterMM = pupilData.diameterMM;
            isCircle = pupilData.isCircle;

            fprintf('  Pupil diameter: %.2f mm\n', estimatedPupilDiameterMM);
            fprintf('  Circularity: %s\n', passFailText(pupilData.isCircle));
            fprintf('  Size validity: %s\n', passFailText(pupilData.isSizeValid));

            % Only continue with reflection analysis if the detected pupil is
            % sufficiently circular and large enough to be considered reliable
            if pupilData.isCircle && pupilData.isSizeValid
                reflectionInput = imbinarize(pupilData.openedImage, ...
                    'adaptive', 'Sensitivity', 0.5);

                % Detect strong bright reflections within the central pupil region
                [hasReflection, reflectionMask] = detectCornealReflectionRobust( ...
                    reflectionInput, ...
                    pupilData.centroid, ...
                    pupilData.radius, ...
                    params.reflectionMinPixelCount, ...
                    params.outerReflectionSearchFactor);

                if hasReflection
                    predictedStatus = "Invalid (Reflection Detected)";
                    overlayPath = fullfile(reflectionVizFolder, ...
                        [baseName '_CR_overlay.bmp']);
                    saveReflectionOverlay(image, reflectionMask, fileName, overlayPath);
                else
                    predictedStatus = "Valid";
                end
            end
        else
            fprintf('  No valid pupil region detected.\n');
        end

        % Save each image into the folder corresponding to its predicted class
        if predictedStatus == "Valid"
            imwrite(image, fullfile(validFolder, fileName));
        else
            imwrite(image, fullfile(invalidFolder, fileName));
        end

        fprintf('  Predicted: %s\n', predictedStatus);

        % If ground-truth labels are available, compute image-level comparison
        if useGroundTruth
            [groundTruth(i), trueStatusString] = ...
                getGroundTruthLabel(fileName, groundTruthTable);

            predictions(i) = double(predictedStatus == "Valid");
            fprintf('  Ground truth: %s (%d)\n', trueStatusString, groundTruth(i));
        end

        % Store image-wise results
        results(i).ImageName = string(fileName);
        results(i).PupilDiameterMM = estimatedPupilDiameterMM;
        results(i).IsCircle = isCircle;
        results(i).PredictedStatus = predictedStatus;
        results(i).HasReflection = hasReflection;
        results(i).ElapsedTimeSec = toc(imageTimer);

        if useGroundTruth
            results(i).TrueStatus = string(trueStatusString);
        end
    end
%% -------------------- Save results --------------------
% Export image-wise processing results to an Excel summary file
resultsTable = struct2table(results);

try
    writetable(resultsTable, excelFileName);
    fprintf('\nProcessing complete.\n');
    fprintf('Results saved to: %s\n', excelFileName);

catch ME
    warning('Could not write results to the default Excel file.');
    fprintf('Reason: %s\n', ME.message);

    try
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        fallbackFile = fullfile(outputFolder, ...
            ['processing_results_backup_' timestamp '.xlsx']);

        writetable(resultsTable, fallbackFile);
        fprintf('Results were saved to backup file:\n%s\n', fallbackFile);

    catch ME2
        warning('Could not write results to the backup Excel file either.');
        fprintf('Backup reason: %s\n', ME2.message);
    end
end
    %% -------------------- Optional evaluation --------------------
    % If ground-truth labels are available, compute classification metrics
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
% CREATEFOLDERIFNEEDED Create output folder if it does not already exist.
%
% Input:
%   folderPath - Full path of the folder to be created
%
% This helper function ensures that required output directories are
% available before saving processed images or results.

    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
end

function txt = passFailText(flag)
% PASSFAILTEXT Convert a logical flag into a readable status string.
%
% Input:
%   flag - Logical value
%
% Output:
%   txt  - 'PASS' if flag is true, otherwise 'FAIL'
%
% This function is used to improve the readability of console messages.

    if flag
        txt = 'PASS';
    else
        txt = 'FAIL';
    end
end

function pupilData = detectPupilRegion(image, params)
% DETECTPUPILREGION Detect the pupil region using morphological processing.
%
% This function estimates the pupil location and geometry from a grayscale
% HS image. The pupil candidate is obtained by applying morphological
% opening, binary thresholding, hole filling, and connected-component
% analysis. The largest connected region is assumed to correspond to the
% pupil.
%
% Inputs:
%   image  - Grayscale input image
%   params - Structure containing processing parameters:
%            .reflectionOpenSE
%            .pixelToMM
%            .eccentricityThreshold
%            .areaDifferenceThreshold
%            .minPupilDiameterMM
%
% Output:
%   pupilData - Structure containing:
%       .found        - True if a pupil candidate was found
%       .centroid     - Estimated pupil centroid [x, y]
%       .radius       - Estimated pupil radius in pixels
%       .diameterMM   - Estimated equivalent pupil diameter in mm
%       .isCircle     - True if the pupil shape passes circularity criteria
%       .isSizeValid  - True if the pupil diameter is above the threshold
%       .openedImage  - Morphologically processed image used later for
%                       reflection detection
%
% Notes:
%   - Circularity is evaluated using eccentricity and area difference
%     relative to an ideal circular approximation.
%   - The equivalent diameter is converted to millimeters using the
%     calibration factor defined in params.pixelToMM.

    pupilData = struct( ...
        'found', false, ...
        'centroid', [NaN, NaN], ...
        'radius', NaN, ...
        'diameterMM', NaN, ...
        'isCircle', false, ...
        'isSizeValid', false, ...
        'openedImage', []);

    % Enhance pupil boundary and suppress small bright structures
    openedImage = imopen(image, params.reflectionOpenSE);

    % Segment dark/bright regions in the processed image
    binaryImage = imbinarize(openedImage);

    % Fill holes to obtain a more complete pupil candidate region
    filledImage = imfill(binaryImage, 'holes');

    % Find all connected regions
    cc = bwconncomp(filledImage);
    if cc.NumObjects == 0
        return;
    end

    % Assume the largest connected component corresponds to the pupil
    numPixels = cellfun(@numel, cc.PixelIdxList);
    [largestSize, idxLargest] = max(numPixels);

    if largestSize <= 0
        return;
    end

    largestComponent = false(size(filledImage));
    largestComponent(cc.PixelIdxList{idxLargest}) = true;

    % Extract geometric properties of the pupil candidate
    props = regionprops(largestComponent, ...
        'Centroid', 'MajorAxisLength', 'Area', 'EquivDiameter', 'Eccentricity');

    if isempty(props)
        return;
    end

    equivDiameter = props.EquivDiameter;
    radius = equivDiameter / 2;
    diameterMM = equivDiameter * params.pixelToMM;

    % Compare area to that of an ideal circle based on the major axis
    majorRadius = props.MajorAxisLength / 2;
    expectedArea = pi * majorRadius^2;

    if expectedArea > 0
        areaDifference = abs(props.Area - expectedArea) / expectedArea;
    else
        areaDifference = inf;
    end

    % Validate the candidate based on circularity and size thresholds
    isCircle = props.Eccentricity < params.eccentricityThreshold && ...
               areaDifference < params.areaDifferenceThreshold;

    isSizeValid = diameterMM >= params.minPupilDiameterMM;

    pupilData.found = true;
    pupilData.centroid = props.Centroid;
    pupilData.radius = radius;
    pupilData.diameterMM = diameterMM;
    pupilData.isCircle = isCircle;
    pupilData.isSizeValid = isSizeValid;
    pupilData.openedImage = openedImage;
end

function [trueLabel, trueStatusString] = getGroundTruthLabel(fileName, groundTruthTable)
% GETGROUNDTRUTHLABEL Retrieve the ground-truth label for a given image.
%
% Inputs:
%   fileName         - Name of the image file
%   groundTruthTable - Table containing at least:
%                      'filename' and 'label'
%
% Outputs:
%   trueLabel        - Numeric label (typically 0 or 1)
%   trueStatusString - Readable label string:
%                      'Valid', 'Invalid', 'Unknown', or 'N/A (Not Found)'
%
% If the image name is not found in the table, the function assigns a
% default invalid label and reports the missing entry.

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

function saveReflectionOverlay(image, reflectionMask, fileName, outputPath)
% SAVEREFLECTIONOVERLAY Save a visualization of detected corneal reflections.
%
% Inputs:
%   image          - Original grayscale image
%   reflectionMask - Binary mask of the detected reflection region
%   fileName       - Name of the processed image
%   outputPath     - Full path for saving the overlay image
%
% This function generates a figure showing the original image with the
% boundaries of the detected reflection region overlaid as dashed red
% contours. The resulting visualization is saved to disk for inspection.

    hFig = figure('Visible', 'off');
    imshow(image);
    hold on;

    boundaries = bwboundaries(reflectionMask, 'noholes');
    for k = 1:numel(boundaries)
        boundary = boundaries{k};
        plot(boundary(:,2), boundary(:,1), 'r--', 'LineWidth', 2);
    end

    title(['CR Detected in ' char(fileName)]);
    hold off;

    frame = getframe(hFig);
    overlayImage = frame2im(frame);
    imwrite(overlayImage, outputPath);
    close(hFig);
end

function evaluateClassification(groundTruth, predictions)
% EVALUATECLASSIFICATION Compute binary classification performance metrics.
%
% Inputs:
%   groundTruth - Vector of true binary labels
%   predictions - Vector of predicted binary labels
%
% This function computes the confusion matrix and the following metrics:
%   - Accuracy
%   - Sensitivity (Recall, True Positive Rate)
%   - Specificity (True Negative Rate)
%   - Precision
%   - F1 Score
%
% The results are displayed in the MATLAB command window.
%
% Notes:
%   - Labels are assumed to follow the convention:
%       1 = Valid
%       0 = Invalid

    if isempty(groundTruth) || isempty(predictions)
        fprintf('\nNot enough data to generate confusion matrix.\n');
        return;
    end

    if numel(groundTruth) ~= numel(predictions)
        error('Length mismatch: groundTruth and predictions must have the same length.');
    end

    C = confusionmat(double(groundTruth), double(predictions), 'Order', [0, 1]);

    TN = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    TP = C(2,2);

    accuracy = (TP + TN) / max(1, TP + TN + FP + FN);
    sensitivity = TP / max(1, TP + FN);
    specificity = TN / max(1, TN + FP);
    precision = TP / max(1, TP + FP);
    f1Score = 2 * precision * sensitivity / max(1, precision + sensitivity);

    fprintf('\n--- Confusion Matrix ---\n');
    disp(C);
    fprintf('Accuracy    : %.4f\n', accuracy);
    fprintf('Sensitivity : %.4f\n', sensitivity);
    fprintf('Specificity : %.4f\n', specificity);
    fprintf('Precision   : %.4f\n', precision);
    fprintf('F1 Score    : %.4f\n', f1Score);
end

function [hasReflection, reflectionImageOut] = detectCornealReflectionRobust( ...
    inputImage, pupilCentroid, pupilRadius, minPixelCountInROI, outerSearchRadiusFactor)
% DETECTCORNEALREFLECTIONROBUST Detect bright corneal reflections inside pupil ROI.
%
% This function searches for bright pixels within a circular region of
% interest centered on the detected pupil. If the number of bright pixels
% inside this region exceeds a minimum threshold, the image is flagged as
% containing a corneal reflection artifact.
%
% Inputs:
%   inputImage               - Binary or logical image highlighting bright
%                              candidate pixels
%   pupilCentroid            - Estimated pupil center [x, y]
%   pupilRadius              - Estimated pupil radius in pixels
%   minPixelCountInROI       - Minimum number of bright pixels required to
%                              confirm a reflection
%   outerSearchRadiusFactor  - Fraction of the pupil radius used to define
%                              the reflection search region
%
% Outputs:
%   hasReflection      - True if a reflection artifact is detected
%   reflectionImageOut - Binary mask containing detected bright pixels
%                        within the search region
%
% Notes:
%   - The search is restricted to a circular region around the pupil center
%     to reduce false detections outside the pupil area.
%   - Input validation is performed to avoid invalid geometric parameters.

    hasReflection = false;
    reflectionImageOut = [];

    if isempty(inputImage) || ...
       numel(pupilCentroid) ~= 2 || ...
       any(isnan(pupilCentroid)) || ...
       pupilRadius <= 0 || ...
       outerSearchRadiusFactor <= 0 || ...
       outerSearchRadiusFactor > 1
        warning('detectCornealReflectionRobust:InvalidInput', ...
            'Invalid input. Returning default values.');
        return;
    end

    centerX = round(pupilCentroid(1));
    centerY = round(pupilCentroid(2));
    outerRadius = max(1, round(pupilRadius * outerSearchRadiusFactor));

    [rows, cols] = size(inputImage);

    minX = max(1, centerX - outerRadius);
    maxX = min(cols, centerX + outerRadius);
    minY = max(1, centerY - outerRadius);
    maxY = min(rows, centerY + outerRadius);

    [Xsub, Ysub] = meshgrid(minX:maxX, minY:maxY);
    distSquared = (Xsub - centerX).^2 + (Ysub - centerY).^2;
    circularMask = distSquared <= outerRadius^2;

    fullMask = false(rows, cols);
    fullMask(minY:maxY, minX:maxX) = circularMask;

    reflectionPixels = false(rows, cols);
    reflectionPixels(fullMask) = inputImage(fullMask);

    if ~any(reflectionPixels(:))
        return;
    end

    brightPixelCount = sum(reflectionPixels(:));
    if brightPixelCount >= minPixelCountInROI
        hasReflection = true;
        reflectionImageOut = reflectionPixels;
    end
end