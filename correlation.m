% Clear workspace
clear;
close all;
clc;
tic;
% --- Configuration: Set your input and output folders ---
% Path to the .mat file containing the Datastores
% NOTE: This path should point to the .mat file, not a folder
matFilePath = 'C:\Ainaimage\16\2 classification\Data_5Folds\Fold_5_Data.mat';
% Path to the output folder for processed and classified images
outputFolder = 'C:\Ainaimage\16\2 classification\correlation\out_test5';

% --- Load the .mat file containing the Datastores ---
disp('Loading the Datastore from .mat file...');
% The 'load' command will create the variables imdsTest and imdsTrain in your workspace
load(matFilePath);

% Ensure imdsTest exists in the loaded data
if ~exist('imdsTest', 'var')
    error('imdsTest variable not found in the specified .mat file.');
end

meanImagePath = 'C:\Ainaimage\17\meanImage_weighted_1024_2.bmp'; % Path to the mean image
correlationThreshold = 0.70; % Threshold for correlation
tolerance = 10; % Tolerance for centering (in pixels)
se_for_reflection_open = strel('disk', 25);
% --- Pupil Diameter Configuration ---
pixelToMM = 0.008833; % IMPORTANT: Replace with your actual calibration factor from pixels to millimeters!
% --- Circle Detection Criteria ---
% eccentricityThreshold = 0.5; % Lower values indicate more circularity (0 is a perfect circle)
% areaDifferenceThreshold = 0.11; % Smaller values mean the actual area is closer to a perfect circle's area

testFileNames = imdsTest.Files;
testLabels = imdsTest.Labels;
numImages = numel(testFileNames);

% Load Ground Truth from Excel file (replace with your file)
groundTruthTable = readtable('C:\Ainaimage\16\2 classification\kfold_circularity\k-10\labels.xlsx');
% فرض می‌شود ستون‌های فایل اکسل 'filename' و 'label' هستند.
groundTruthFileNames = groundTruthTable.filename; 
groundTruthLabels = groundTruthTable.label; 

% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
validFolder = fullfile(outputFolder, 'Valid');
if ~exist(validFolder, 'dir'), mkdir(validFolder); end
invalidFolder = fullfile(outputFolder, 'inValid');
if ~exist(invalidFolder, 'dir'), mkdir(invalidFolder); end

% Read the mean image
meanImage = imread(meanImagePath);
meanImage = double(meanImage); % Convert to double for calculations

% Initialize arrays for storing results for Excel export
excelData = cell(numImages, 6); 
excelHeader = {'File Name', 'Estimated Pupil Diameter (mm)','Correlation Coefficient', 'Actual Label', 'Predicted Label', 'Elapsed Time (s)'};

% Initialize arrays for Ground Truth and Predictions for Confusion Matrix
allGroundTruthNumeric = zeros(numImages, 1);
allPredictionsNumeric = zeros(numImages, 1);

% Process each image
for i = 1:numImages
    tic % Start timer for current image
    I_original_path = testFileNames{i};
    [I_original, ~] = readimage(imdsTest, i); % Use readimage for a Datastore
    
    [~, fileName, ext] = fileparts(I_original_path);
    fileName = [fileName ext]; % Recreate the original filename
    
    fprintf('\n--- Processing image: %s ---\n', fileName);
    
    % Initialize variables for this image
    estimatedPupilDiameterMM_Equiv = NaN; % Initialize to NaN
    correlationCoefficient = NaN; % Initialize correlation coefficient
    isCentered = false; % Assume not centered until proven otherwise
    isCircle = false; % Assume not circular until proven otherwise
    
    % Default prediction status (0 for Invalid, 1 for Valid)
    prediction_status_numeric = 0; 
    predictedLabelStr = 'Invalid'; % Default prediction for Excel output
    outputDestinationFolder = invalidFolder; % Default output folder
    
    % Prepare image for pupil detection
    opened_closed_spots = imopen(I_original, se_for_reflection_open);
    binaryImage = imbinarize(opened_closed_spots);
    final_pupil_region = imfill(binaryImage, 'holes');
    
    % --- Find the largest connected component (assumed to be the pupil) ---
    CC = bwconncomp(final_pupil_region); % Finds connected components
    numPixels = cellfun(@numel, CC.PixelIdxList); % Counts pixels in each component
    largest_component = false(size(final_pupil_region)); % Initialize largest_component
    
    currentImageForCorrelation = double(I_original); % Default to original for correlation
    
    if ~isempty(numPixels) && max(numPixels) > 0 % Check if any connected components were found
        [~, idx] = max(numPixels); % Finds the size and index of the largest component
        largest_component(CC.PixelIdxList{idx}) = true; % Creates a binary image of only the largest component
    else
        fprintf('  No connected components found for image: %s. Skipping pupil property extraction.\n', fileName);
        % If no pupil is detected, it remains 'Invalid' (prediction_status_numeric = 0).
    end
    
    % Find the centroid of the largest region AND extract EquivDiameter
    props = regionprops(largest_component,'Centroid','MajorAxisLength', 'MinorAxisLength','Area', 'EquivDiameter','Eccentricity');
    
    if ~isempty(props) && props(1).Area > 0 % Check if valid properties were extracted and area is positive
        
        % --- Calculate Pupil Diameter ---
        estimatedPupilDiameterMM_Equiv = props(1).EquivDiameter * pixelToMM; % Use props(1) as it's the largest
        fprintf('  Estimated Pupil Diameter: %.2f mm\n', estimatedPupilDiameterMM_Equiv);
        
               
        % --- Check and Correct Centering ---
        [rows, cols] = size(I_original);
        imageCenterX = round(cols / 2);
        imageCenterY = round(rows / 2);
        centroidX = round(props(1).Centroid(1));
        centroidY = round(props(1).Centroid(2));
        
        % Check if the centroid is within the tolerance
        if abs(centroidX - imageCenterX) <= tolerance && abs(centroidY - imageCenterY) <= tolerance
            isCentered = true;
            fprintf('  - Image is already centered within tolerance.\n');
            currentImageForCorrelation = double(I_original); % Use original image (as it is already centered)
        else
            % Translate the image to center it
            deltax = imageCenterX - centroidX;
            deltay = imageCenterY - centroidY;
            currentImageForCorrelation = imtranslate(I_original, [deltax, deltay], 'FillValues', 0);
            currentImageForCorrelation = double(currentImageForCorrelation); % Convert to double for correlation
            fprintf('  - Image translated to center (dx: %d, dy: %d).\n', deltax, deltay);
            isCentered = true; % Set to true after attempting to center
        end
    else % No valid properties found
        fprintf('  No valid pupil properties found after morphology.\n');
        % isCentered, isCircle remain false as initialized (prediction_status_numeric = 0)
    end
    
    % --- Core Logic for Classification ---
    if isCentered   % Only proceed if the pupil is found, is circular, and can be centered
        
        % Calculate correlation on the (potentially) centered image
        meanVector = meanImage(:);
        currentVector = currentImageForCorrelation(:);
        
        % اطمینان از اینکه ابعاد دو بردار یکسان است
        if length(meanVector) == length(currentVector)
            correlationCoefficient = corr(meanVector, currentVector);
            fprintf('  Correlation Coefficient: %.4f\n',  correlationCoefficient);
            
            if correlationCoefficient >= correlationThreshold
                
                % Check for reflection (on the original image)
                reflection = detectReflection(I_original); 
                
                if reflection
                    prediction_status_numeric = 0; % Invalid due to reflection
                    outputDestinationFolder = invalidFolder;
                    fprintf('  Image is valid by correlation but has reflection. Saved to invalid folder.\n');
                else
                    prediction_status_numeric = 1; % Valid 
                    outputDestinationFolder = validFolder;
                    predictedLabelStr = 'Valid'; 
                    fprintf('  Image is valid by all criteria. Saved to valid folder.\n');
                end
            else % Correlation is too low
                prediction_status_numeric = 0; % Invalid due to low correlation
                outputDestinationFolder = invalidFolder;
                fprintf('  Image is invalid (low correlation). Saved to invalid folder.\n');
            end
        else
            fprintf('  ERROR: Image dimensions do not match mean image. Correlation skipped.\n');
        end
    else % Not centered OR Not circular OR No pupil found
         prediction_status_numeric = 0; % Invalid
         outputDestinationFolder = invalidFolder;
         fprintf('  Image is invalid (pupil not found/not circular/could not be centered).\n');
    end
    
    % --- Find the ground truth label and store for confusion matrix calculation ---
    groundTruthIndex = find(strcmp(groundTruthFileNames, fileName), 1);
    if ~isempty(groundTruthIndex)
        currentTrueStatusValue = groundTruthLabels(groundTruthIndex);
        currentGroundTruthNumeric = currentTrueStatusValue; 
        if currentTrueStatusValue == 1
            currentTrueStatusString = 'Valid';
        elseif currentTrueStatusValue == 0
            currentTrueStatusString = 'Invalid';
        else
            currentTrueStatusString = 'Unknown';
        end
        fprintf('  True status from Excel: %s (%d)\n', currentTrueStatusString, currentTrueStatusValue);
    else
        currentGroundTruthNumeric = 0; % Default to 0 (Invalid) if not found
        currentTrueStatusString = 'N/A (Not Found)';
        fprintf('  Warning: True label for %s not found. Defaulting to Invalid (0).\n', fileName);
    end
    
    % --- Save the original image to the appropriate folder (Valid/Invalid) ---
    if prediction_status_numeric == 1
        outputFilename = fullfile(validFolder, fileName);
    else
        outputFilename = fullfile(invalidFolder, fileName);
    end
    imwrite(I_original, outputFilename); % ذخیره تصویر اصلی (I_original)
    
    elapsedTime = toc; % End timer for current image
   
    % Store data for Excel export
      excelData{i, 1} = fileName;
    excelData{i, 2} = estimatedPupilDiameterMM_Equiv; 
    excelData{i, 3} = correlationCoefficient; 
    excelData{i, 4} = currentTrueStatusString;
    excelData{i, 5} = predictedLabelStr;
    excelData{i, 6} = elapsedTime;
    
    % Store numeric ground truth and prediction for confusion matrix
    allGroundTruthNumeric(i) = currentGroundTruthNumeric;
    allPredictionsNumeric(i) = prediction_status_numeric; 
end

% Create a table from the cell array
resultsTable = cell2table(excelData, 'VariableNames', excelHeader);
% Define the output Excel filename
excelOutputFilename = fullfile(outputFolder, 'PupilAnalysisResults5.xlsx');
% Write the table to an Excel file
writetable(resultsTable, excelOutputFilename);
fprintf('\nAll detailed results exported to %s\n', excelOutputFilename);

% --- Calculate and Display Confusion Matrix and Metrics ---
disp('--- Overall Classification Performance ---');

% Conversion of allGroundTruthNumeric and allPredictionsNumeric to categorical or string
% The confusionmat function in recent MATLAB versions handles numeric arrays, 
% but ensure they only contain the expected classes (0 and 1).
unique_classes = unique([allGroundTruthNumeric; allPredictionsNumeric]);

% Ensure all elements are 0 or 1.
if isempty(unique_classes) || any(~ismember(unique_classes, [0 1]))
    disp('Warning: Ground Truth or Prediction contains unexpected class labels. Skipping Confusion Matrix calculation.');
else
    % Re-calculate for robustness: assuming class 1 is 'Valid' (Positive) and 0 is 'Invalid' (Negative)
    C = confusionmat(allGroundTruthNumeric, allPredictionsNumeric, 'Order', [0 1]); 

     
    TP = C(1,1);
    FN = C(1,2);
    FP = C(2,1);
    TN = C(2,2);
    
    accuracy = (TP + TN) / sum(C(:));
    sensitivity = TP / (TP + FN); % Recall for Valid (Positive)
    specificity = TN / (TN + FP); % Recall for Invalid (Negative)
    precision = TP / (TP + FP); % Precision for Valid (Positive)
    
    if (precision + sensitivity) == 0
        f1Score = 0;
    else
        f1Score = 2 * (precision * sensitivity) / (precision + sensitivity);
    end
    
    % disp('Confusion Matrix (Rows=Actual, Columns=Predicted; 1=Valid, 0=Invalid):');
    disp(C);
    disp(['Accuracy: ', num2str(accuracy)]);
    disp(['Sensitivity: ', num2str(sensitivity)]);
    disp(['Specificity: ', num2str(specificity)]);
    disp(['Precision : ', num2str(precision)]);
    disp(['F1-score: ', num2str(f1Score)]);
end
fprintf('Processing complete.\n');

% --- Helper Function for Reflection Detection ---
% This function looks for bright circular objects which could be reflections
function reflection = detectReflection(image)
        % اطمینان از اینکه ورودی grayscale است
        if size(image, 3) == 3
             grayImage = rgb2gray(image);
        else
             grayImage = image;
        end
        % imfindcircles نیاز به تصویر uint8 دارد
        if ~isa(grayImage, 'uint8')
            grayImage = im2uint8(grayImage);
        end
   
    [centers, ~, ~] = imfindcircles(grayImage, [20 90], 'ObjectPolarity', 'bright', 'Sensitivity', 0.9); % Tuned range for reflection
    reflection = ~isempty(centers); % If any bright circles are found, assume reflection
end