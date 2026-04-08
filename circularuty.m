clear;
close all;
clc;
tic;

% --- Configuration: Set your input and output folders ---
% Path to the .mat file containing the Datastores
% NOTE: This path should point to the .mat file, not a folder
matFilePath = 'C:\Ainaimage\16\2 classification\Data_5Folds\Fold_1_Data.mat';

% Path to the output folder for processed and classified images
outputFolder = 'C:\Ainaimage\16\2 classification\kfold_circularity\k-5\out_blink11';

% --- Load the .mat file containing the Datastores ---
disp('Loading the Datastore from .mat file...');
% The 'load' command will create the variables imdsTest and imdsTrain in your workspace
load(matFilePath);

% Ensure imdsTest exists in the loaded data
if ~exist('imdsTest', 'var')
    error('imdsTest variable not found in the specified .mat file.');
end

testFileNames = imdsTest.Files;
% testLabels = imdsTest.Labels;
numImages = numel(testFileNames);

groundTruthTable = readtable('C:\Ainaimage\16\2 classification\kfold_circularity\k-10\labels.xlsx');
groundTruthFileNames = groundTruthTable.filename;
groundTruthLabels = groundTruthTable.label;

pixelToMM = 0.008833; % 0.008833 millimeters per pixel
minPupilDiameterMM = 3.0; % pupil diameter threshold in mm
reflection_min_pixel_count_in_roi = 1000; % Minimum number of bright pixels in the ROI for a valid reflection detection
outer_reflection_search_factor = 0.6; % Adjust this value (0 to 1)
se_for_reflection_open = strel('disk', 25);

% Excel filename to save processing results
excelFileName = fullfile(outputFolder, 'processing_results_circularity10.xlsx');
% Folder name for saving reflection visualizations (optional)
reflectionVizFolder = fullfile(outputFolder, 'Reflection_Visualizations');

% --- Ensure output folders exist, create them if not ---
if ~exist(outputFolder, 'dir'), mkdir(outputFolder); end
validFolder = fullfile(outputFolder, 'Valid'); % Folder for 'Valid' images
if ~exist(validFolder, 'dir'), mkdir(validFolder); end
invalidFolder = fullfile(outputFolder, 'inValid'); % Folder for 'Invalid' images
if ~exist(invalidFolder, 'dir'), mkdir(invalidFolder); end
if ~exist(reflectionVizFolder, 'dir'), mkdir(reflectionVizFolder); end % Create folder for reflection visualizations

% --- Prepare Excel header and results storage ---
header = {'Image Name', 'Pupil Diameter (mm)', 'Is Circle?', 'True Status', 'Predicted Status', 'Has Reflection', 'Elapsed Time (s)'};
results = cell(0, length(header)); % Empty cell array to store results

% --- Initialize arrays for Confusion Matrix ---
groundTruth = zeros(numImages, 1); % Actual labels (from Datastore)
predictions = zeros(numImages, 1); % Predicted labels (by algorithm)

% --- Main loop for processing each image from the Datastore ---
for i = 1:numImages
    tic % Start timer for current image processing

    % --- Read image and label from the Datastore ---
    I_original_path = testFileNames{i};
    [I_original, info] = readimage(imdsTest, i); % Use readimage for a Datastore
    
    [~, fileName, ext] = fileparts(I_original_path);
    fileName = [fileName ext]; % Recreate the original filename
    
    % fprintf('\n--- Processing image: %s ---\n', fileName);
    
    % Initialize variables for the current image
    estimatedPupilDiameterMM_Equiv = NaN;
    predictedStatus = 'Undefined';
    isCircle = false;
    isPupilSizeValid = false;
    hasReflection = false; % Flag for reflection detection

    % Convert to grayscale if the image is color
    if size(I_original, 3) == 3
        I_original = rgb2gray(I_original);
    end
 
    % --- 2. Pupil Detection Steps (Morphology-based) ---
    opened_closed_spots = imopen(I_original, se_for_reflection_open);
    binaryImage = imbinarize(opened_closed_spots);
    final_pupil_region = imfill(binaryImage, 'holes');
    
    CC = bwconncomp(final_pupil_region);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    pupilCentroid = [NaN, NaN];
    pupilRadius = NaN;
    
    [biggest_size, idx] = max(numPixels);
    
    if biggest_size > 0
        largest_component = false(size(final_pupil_region));
        largest_component(CC.PixelIdxList{idx}) = true;
        props = regionprops(largest_component, 'Centroid', 'MajorAxisLength', 'Area', 'EquivDiameter', 'Eccentricity');
        
        if isempty(props)
            continue
        end
        
        pupilCentroid = props.Centroid;
        pupilRadius = props.EquivDiameter / 2;
        estimatedPupilDiameterMM_Equiv = props.EquivDiameter * pixelToMM;
        
        fprintf('    Equivalent Pupil Diameter: %.2f %s\n', estimatedPupilDiameterMM_Equiv, 'mm');
        
        eccentricityThreshold = 0.5;
        areaDifferenceThreshold = 0.11;
        currentEccentricity = props.Eccentricity;
        expectedRadius = props.MajorAxisLength / 2;
        expectedArea = pi * (expectedRadius^2);
        areaDifference = inf;
        if expectedArea > 0
            areaDifference = abs(props.Area - expectedArea) / expectedArea;
        end
        
        if (currentEccentricity < eccentricityThreshold) && (areaDifference < areaDifferenceThreshold)
            isCircle = true;
            fprintf('  - Object identified as circular (Eccentricity: %.2f, Area Diff: %.2f%%)\n', currentEccentricity, areaDifference * 100);
        else
            isCircle = false;
            fprintf('  - Object not identified as circular (Eccentricity: %.2f, Area Diff: %.2f%%)\n', currentEccentricity, areaDifference * 100);
        end
        
        if estimatedPupilDiameterMM_Equiv >= minPupilDiameterMM
            isPupilSizeValid = true;
            fprintf('  - Pupil diameter (%.2f mm) is valid.\n', estimatedPupilDiameterMM_Equiv);
        else
            isPupilSizeValid = false;
            fprintf('  - Pupil diameter (%.2f mm) is too small (< %.2f mm). Invalidated.\n', estimatedPupilDiameterMM_Equiv, minPupilDiameterMM);
        end
        
        % --- Logic for determining final Predicted Status ---
        if  isempty(props) || ~isCircle || ~isPupilSizeValid
            predictedStatus = 'Invalid';
            predictions(i) = 0;
        else
            b_for_reflection_detection = imbinarize(opened_closed_spots, 'adaptive', 'Sensitivity', 0.5);
            [hasReflection,reflectionImage_out] = detectCornealReflection_robust(b_for_reflection_detection, pupilCentroid, pupilRadius, reflection_min_pixel_count_in_roi, outer_reflection_search_factor);
            
            if hasReflection
                predictedStatus = 'Invalid (Reflection Detected)';
                predictions(i) = 0;
                [~, name_only, ~] = fileparts(fileName);
                outputFilenameOverlay = fullfile(reflectionVizFolder, [name_only '_CR_overlay.bmp']);
                
                % Display and Save Overlaid Image
                hFig_overlay = figure('Visible', 'off');
                imshow(I_original);
                hold on;
                B = bwboundaries(reflectionImage_out, 'noholes');
                for k = 1:length(B)
                    boundary = B{k};
                    plot(boundary(:, 2), boundary(:, 1), 'r--', 'LineWidth', 2);
                end
                title(['CR Detected in ' fileName]);
                hold off;
                F = getframe(hFig_overlay);
                Image = frame2im(F);
                imwrite(Image, outputFilenameOverlay);
                close(hFig_overlay);
            else
                predictedStatus = 'Valid';
                predictions(i) = 1;
            end
        end
    else
        predictedStatus = 'Invalid';
        predictions(i) = 0;
    end
        elapsedTime = toc;

    fprintf('  --> Image %s classified as: %s (predictions(i) = %d)\n', fileName, predictedStatus, predictions(i));
    
    % --- Get True Status from Datastore and populate Ground Truth array ---
     groundTruthIndex = find(strcmp(groundTruthTable.filename, fileName), 1);
    if ~isempty(groundTruthIndex)
        currentTrueStatusValue = groundTruthTable.label(groundTruthIndex);
        groundTruth(i) = currentTrueStatusValue;
        if currentTrueStatusValue == 1
            currentTrueStatusString = 'Valid';
        elseif currentTrueStatusValue == 0
            currentTrueStatusString = 'Invalid';
        else
            currentTrueStatusString = 'Unknown';
        end
        fprintf('  True status from Excel: %s (%d)\n', currentTrueStatusString, currentTrueStatusValue);
    else
        % If true label not found, default to "Invalid" (0).
        groundTruth(i) = 0;
        currentTrueStatusString = 'N/A (Not Found)';
        fprintf('  Warning: True label for %s not found. Defaulting to Invalid (0).\n', fileName);
    end
    
    % --- Save the original image to the appropriate folder (Valid/Invalid) ---
    if strcmp(predictedStatus, 'Valid')
        outputFilename = fullfile(validFolder, fileName);
        imwrite(I_original, outputFilename);
    else
        outputFilename = fullfile(invalidFolder, fileName);
        imwrite(I_original, outputFilename);
    end
    
    % elapsedTime = toc;
    results = [results; {fileName, estimatedPupilDiameterMM_Equiv, isCircle, currentTrueStatusString, char(predictedStatus), hasReflection, elapsedTime}];
end

% --- Write all collected results to the Excel file ---
resultsTable = cell2table(results, 'VariableNames', header);
writetable(resultsTable, excelFileName);
fprintf('\nProcessing complete. Results saved to: %s\n', excelFileName);

% --- Generate Confusion Matrix and Classification Metrics ---
if numImages > 0 && ~isempty(predictions) && ~isempty(groundTruth)
    fprintf('\n--- Generating Confusion Matrix ---\n');
    groundTruth = double(groundTruth);
    predictions = double(predictions);
    
    if length(groundTruth) ~= length(predictions)
        error('Error: Length of groundTruth (%d) does not match length of predictions (%d). Cannot compute confusion matrix.', length(groundTruth), length(predictions));
    end
    
    labels_to_evaluate = [0, 1];
    C = confusionmat(groundTruth, predictions, 'Order', labels_to_evaluate);
    
    TN = C(1,1);
    FP = C(1,2);
    FN = C(2,1);
    TP = C(2,2);
    
    accuracy = (TP + TN) / max(1, (TP + TN + FP + FN));
    sensitivity = TP / max(1, (TP + FN));
    specificity = TN / max(1, (TN + FP));
    precision = TP / max(1, (TP + FP));
    f1Score = 2 * (precision * sensitivity) / max(1, (precision + sensitivity));
    
    disp('Confusion Matrix:');
    disp(C);
    disp(['Accuracy: ', num2str(accuracy)]);
    disp(['Sensitivity: ', num2str(sensitivity)]);
    disp(['Specificity: ', num2str(specificity)]);
    disp(['Precision: ', num2str(precision)]);
    disp(['F1 Score: ', num2str(f1Score)]);
else
    fprintf('\nNot enough data to generate confusion matrix (ground truth or predictions are empty).\n');
end

% --- Function to detect Corneal Reflection (CR) ---
function [hasReflection,reflectionImage_out] = detectCornealReflection_robust(inputImage, pupilCentroid_in, pupilRadius_in, min_pixel_count_in_roi, outer_search_radius_factor)
% The function is the same as the original.
hasReflection = false;
reflectionImage_out = [];
if isempty(inputImage) || numel(pupilCentroid_in) ~= 2 || any(isnan(pupilCentroid_in)) || pupilRadius_in <= 0 || outer_search_radius_factor <= 0 || outer_search_radius_factor > 1
    warning('detectCornealReflection_robust:InvalidInput', 'Invalid input to detectCornealReflection_robust. Check pupil parameters and search factors. Returning default values.');
    return;
end
centerX = round(pupilCentroid_in(1));
centerY = round(pupilCentroid_in(2));
outerRadius = round(pupilRadius_in * outer_search_radius_factor);
if outerRadius < 1
    outerRadius = 1;
end
[rows, cols] = size(inputImage);
minX = max(1, centerX - outerRadius);
maxX = min(cols, centerX + outerRadius);
minY = max(1, centerY - outerRadius);
maxY = min(rows, centerY + outerRadius);
[X_sub, Y_sub] = meshgrid(minX:maxX, minY:maxY);
distSquared_sub = (X_sub - centerX).^2 + (Y_sub - centerY).^2;
circle_mask_sub = (distSquared_sub <= outerRadius^2);
full_mask = false(rows, cols);
full_mask(minY:maxY, minX:maxX) = circle_mask_sub;
binary_bright_pixels_in_circle = false(rows, cols);
circle_pixels = inputImage(full_mask);
binary_bright_pixels_in_circle(full_mask) = circle_pixels;
if ~any(binary_bright_pixels_in_circle(:))
    return;
end
all_ones_in_binary_bright_pixels_in_circle = sum(binary_bright_pixels_in_circle(:));
if all_ones_in_binary_bright_pixels_in_circle >= min_pixel_count_in_roi
    hasReflection = true;
    reflectionImage_out = binary_bright_pixels_in_circle;
end
return;
end