
folderName = 'C:\Ainaimage\16\4 classification\InputData';

% Create an imageDatastore to manage the images
imds = imageDatastore(folderName, ...
    'IncludeSubfolders', true, ... % Include images in all subfolders
    'LabelSource', "foldernames", ... % Use folder names as class labels
    'ReadFcn', @readAndConvert); % Use a custom function to read/preprocess images

% Define the number of folds for cross-validation (e.g., 10-fold CV)
numFolds = 5;

% Array to store the accuracy for each fold for the first model (ShuffleNet)
Accuracy = zeros(numFolds, 1);

% Get the expected input size from the first layer of the pre-defined network architecture
inputSize = net_1.Layers(1).InputSize;

% Determine the number of classes based on the image labels
numClasses = numel(categories(imds.Labels));

% Loop to perform the fold cross-validation
for i = 1:numFolds
    
    % Load the pre-saved training and testing data for the current fold
    filename = sprintf('Fold_%d_Data.mat', i);
    load(filename, 'imdsTrain', 'imdsTest');
    
    fprintf('Starting training and evaluation for Fold %d of %d...\n', i, numFolds);
    
    % --- Training and Evaluation for ShuffleNet (or any other model) ---
     imdsTrain.ReadFcn = @readAndConvert;
    imdsTest.ReadFcn = @readAndConvert;
    % Create an augmentedImageDatastore for the current fold's data
    % Note: 'imdsTrain' and 'imdsTest' are loaded from the file.
    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
    augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);
    
    % Set the training options
   options = trainingOptions("adam", ... % Use the Adam optimizer
        MaxEpochs=40, ... % Maximum number of epochs
        Plots="training-progress", ...% Display the training progress plot
        InitialLearnRate=0.0001,....
        Metrics="accuracy", ... % Track accuracy during training
        Verbose=false, ... % Suppress command window output during training
        ExecutionEnvironment="gpu"); % Utilize the GPU for faster training
        
    % Train the model
    % It is assumed that 'ShuffleNet_2class' is a pre-defined network architecture.
    net = trainnet(augimdsTrain, net_1, "crossentropy", options);
    
    % Predict on the test data using mini-batches
    YPred = minibatchpredict(net, augimdsTest);
    
    % Convert the raw predictions (scores) into predicted labels
    [~, predictedLabels] = max(YPred, [], 2);
    % Map the numerical indices back to categorical labels
    YPred_Categorical = categorical(predictedLabels, 1:numClasses, categories(imds.Labels));
    
    % Get the true labels for the test set
    YTest = imdsTest.Labels;
    
    % Display the confusion matrix for the current fold's results
    figure
    confusionchart(YTest, YPred_Categorical); 
    
    % Calculate the fold accuracy and store it
    accuracy = mean(YPred_Categorical == YTest);
    Accuracy(i) = accuracy;
    
    fprintf('Completed Fold %d. Test Accuracy: %.2f%%\n', i, accuracy * 100);
    
end

% --- Calculate and Display the Final Mean Accuracy ---
meanAccuracy = mean(Accuracy);
stdAccuracy = std(Accuracy); % Calculate the standard deviation of the accuracies
fprintf('\n----------------------------------------\n');
fprintf('Average 5-Fold Accuracy : %.2f%% (+/- %.2f%%)\n', meanAccuracy * 100, stdAccuracy * 100);
fprintf('----------------------------------------\n');

%=========================time==================================================
% 
% % Define the folder path for the images used in timing the prediction
% testFolder = 'C:\Ainaimage\16\2 classification\Data-Fold-1';
% 
% % Create an imageDatastore to read all images in the test folder
% imdsTestFolder = imageDatastore(testFolder);
% 
% % Total number of images to be tested
% numImagesToTest = numel(imdsTestFolder.Files);
% 
% % Create an array to store the execution time for each image
% executionTimes = zeros(numImagesToTest, 1);
% 
% % Transfer the trained model to the CPU for more accurate time calculation (if necessary)
% net_cpu = gather(net); 
% 
% % Start the loop to process (predict on) each image
% for i = 1:numImagesToTest
% 
%     % Read the current image from the datastore
%     im = readimage(imdsTestFolder, i);
% 
%     % Start the timer (tic)
%     tic;
% 
%     % Pre-process the image (as per your code)
%     % Convert grayscale to 3-channel (RGB) if necessary
%     if size(im, 3) == 1
%         im = cat(3, im, im, im);
%     end
%     % Resize the image to the network's required input size
%     im = imresize(im, inputSize(1:2));
%     % Convert the image to single precision for network input
%     X = single(im);
% 
%     % Execute the prediction. (No need to store the output, only the time matters)
%     predict(net_cpu, X);
% 
%     % Stop the timer (toc) and store the elapsed time in the array
%     executionTimes(i) = toc;
% end
% 
% % Calculate the average execution time across all images
% averageTime = mean(executionTimes);
% 
% % Display the final result
% fprintf('==========================================================\n');
% fprintf('Average prediction time for %d images: %.4f seconds\n', numImagesToTest, averageTime);
% fprintf('==========================================================\n');