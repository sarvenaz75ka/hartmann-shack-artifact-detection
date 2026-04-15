function create_kfold_datastores(dataFolder, outputFoldFolder, numFolds)
% CREATE_KFOLD_DATASTORES
%
% Create stratified k-fold train/test datastores from an image dataset.
%
% The dataset must be organized in subfolders, where each subfolder
% corresponds to one class label.
%
% Usage:
%   create_kfold_datastores(dataFolder, outputFoldFolder)
%   create_kfold_datastores(dataFolder, outputFoldFolder, numFolds)
%
% Inputs:
%   dataFolder        - Root folder containing labeled images
%   outputFoldFolder  - Folder where Fold_1_Data.mat, Fold_2_Data.mat, ...
%                       will be saved
%   numFolds          - Number of folds (default: 5)
%
% Output:
%   outputFoldFolder/
%       ├── Fold_1_Data.mat
%       ├── Fold_2_Data.mat
%       ├── ...
%       └── Fold_k_Data.mat
%
% Notes:
%   - Stratified partitioning is used to preserve class balance
%   - Each MAT file contains:
%       imdsTrain
%       imdsTest

    if nargin < 2
        error('You must provide dataFolder and outputFoldFolder.');
    end

    if nargin < 3 || isempty(numFolds)
        numFolds = 5;
    end

    if ~isfolder(dataFolder)
        error('Data folder does not exist: %s', dataFolder);
    end

    if ~exist(outputFoldFolder, 'dir')
        mkdir(outputFoldFolder);
    end

    % Read full dataset
    imds = imageDatastore(dataFolder, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    if isempty(imds.Files)
        error('No images were found in the data folder.');
    end

    labels = imds.Labels;

    % Stratified K-fold partition
    cv = cvpartition(labels, 'KFold', numFolds);

    fprintf('Creating %d-fold datastore partitions...\n', numFolds);

    for foldIdx = 1:numFolds
        trainIdx = training(cv, foldIdx);
        testIdx = test(cv, foldIdx);

        imdsTrain = subset(imds, trainIdx);
        imdsTest = subset(imds, testIdx);

        foldFile = fullfile(outputFoldFolder, sprintf('Fold_%d_Data.mat', foldIdx));
        save(foldFile, 'imdsTrain', 'imdsTest');

        fprintf('Saved Fold %d: %d train / %d test images -> %s\n', ...
            foldIdx, numel(imdsTrain.Files), numel(imdsTest.Files), foldFile);
    end

    fprintf('All folds were created successfully.\n');
end