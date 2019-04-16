url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder
if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end
rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'airplanes', 'ferry', 'laptop'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
[trainingSet, validationSet] = splitEachLabel(imds, 0.3, 'randomize');
bag = bagOfFeatures(trainingSet);
trainFeatures = encode(bag, trainingSet);
SVM_SURF = fitcecoc(trainFeatures,trainingSet.Labels);
featureMatrix = encode(bag, validationSet);
[pred, score, cost] = predict(SVM_SURF, featureMatrix);
accuracy = sum(validationSet.Labels == pred)/size(validationSet.Labels,1);
accuracy;