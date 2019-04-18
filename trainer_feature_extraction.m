function [svm_model, knn_model] = trainer_feature_extraction()
    %% Image training and finding eigenfaces
    imagelist = dir('image_set');
    noOfFile = length(imagelist);
    noOfImage = (noOfFile - 2);
    imagelist = imagelist(3:noOfFile);
    disp(noOfImage);
    imageSet = cell(1, noOfImage);
    HEset = cell(1, noOfImage);
    LOGset = cell(1, noOfImage);
    current_dir = replace(pwd, '\', '/');
    image_dir =  '/image_set/';
    test_image_dir = '/test_image/';
    disp(current_dir);
    avg_dim = 0;
    vectorized_images = zeros(10000, noOfImage);
    he_images = zeros(10000, noOfImage);
    size_mat = 4356;
    feature_matrix = zeros(noOfImage*7, size_mat);
    image_class_matrix = cell(noOfImage*7,1);
    %f_matrix = zeros(noOfImage, 4356);

    %%% read each image and store it in the set of matrix
    %%% Vectorize the image matrix
    %%
    figure;
    for i = 1:noOfImage
        filename = strcat(current_dir,image_dir,imagelist(i).name);
        resizedImage = imresize(rgb2gray(detectFace(imread(filename))),[100 100]);
        imageSet{i} = resizedImage;
        %vectorized_images(:, i) = reshape(double(resizedImage),[], 1);
        I = resizedImage;
        %Edge detection of my face from the training database
        [~, threshold] = edge(I, 'sobel');
        fudgeFactor = .7;
        BW1 = edge(I,'sobel', threshold * fudgeFactor);

        %extract the hog features of the processed face
        [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
        %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
        feature_matrix(i,:) = featureVector;

        % specifying classs
        [filepath,name,ext] = fileparts(filename);
        %disp(name);
        name = split(name, '-');
        image_class_matrix{i} = name{1};

        %% histogram eq
        im = histeq(resizedImage);

        %HEset{i} = im;
        %he_images(:,i) = reshape(double(im),[],1);
        %vectorized_images(:, i+25) = reshape(double(im),[],1);
        imageSet{i+noOfImage} = im;
        %Edge detection of my face from the training database
        [~, threshold] = edge(im, 'sobel');
        fudgeFactor = .7;
        BW1 = edge(im,'sobel', threshold * fudgeFactor);

        %extract the hog features of the processed face
        [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
        %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
        feature_matrix(i+noOfImage,:) = featureVector;

        % histeq
        image_class_matrix{i+noOfImage} = name{1};

        %% gaussian noise
        im = histeq(resizedImage);

        %HEset{i} = im;
        %he_images(:,i) = reshape(double(im),[],1);
        %vectorized_images(:, i+25) = reshape(double(im),[],1);
        imageSet{i+ noOfImage + noOfImage} = imnoise(im, 'gaussian');
        %Edge detection of my face from the training database
        [~, threshold] = edge(im, 'sobel');
        BW1 = edge(im,'sobel', threshold * fudgeFactor);

        %extract the hog features of the processed face
        [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
        %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
        feature_matrix(i+noOfImage+noOfImage,:) = featureVector;

        %gaussian
        image_class_matrix{i+noOfImage+noOfImage} = name{1};


        %% rotate Image 7 degreee

        im = imrotate(resizedImage, 5,'bilinear', 'crop' );
        %Edge detection of my face from the training database
        [~, threshold] = edge(im, 'sobel');
        BW1 = edge(im,'sobel', threshold * fudgeFactor);

        %extract the hog features of the processed face
        [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
        %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
        feature_matrix(i+(noOfImage*3),:) = featureVector;

        %rotated +7
        image_class_matrix{i+(noOfImage*3)} = name{1};

        % rotated image -7 degree
        %% rotate Image -7 degree

        im = imrotate(resizedImage, -5,'bilinear', 'crop' );
        %Edge detection of my face from the training database
        [~, threshold] = edge(im, 'sobel');
        BW1 = edge(im,'sobel', threshold * fudgeFactor);

        %extract the hog features of the processed face
        [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
        %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
        feature_matrix(i+(noOfImage*4),:) = featureVector;

        image_class_matrix{i+(noOfImage*4)} = name{1};
        % rotated -7

        %% adapt his equal
        im = adapthisteq(resizedImage,'NumTiles',[8 8],'ClipLimit',0.005);
        %Edge detection of my face from the training database
        [~, threshold] = edge(im, 'sobel');
        BW1 = edge(im,'sobel', threshold * fudgeFactor);

        %extract the hog features of the processed face
        [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
        %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
        feature_matrix(i+(noOfImage*5),:) = featureVector;

        image_class_matrix{i+(noOfImage*5)} = name{1};

        %% rotate Image 10 degreee

        im = imrotate(resizedImage, 10,'bilinear', 'crop' );
        %Edge detection of my face from the training database
        [~, threshold] = edge(im, 'sobel');
        BW1 = edge(im,'sobel', threshold * fudgeFactor);

        %extract the hog features of the processed face
        [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
        %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
        feature_matrix(i+(noOfImage*6),:) = featureVector;

        %rotated +7
        image_class_matrix{i+(noOfImage*6)} = name{1};

        % rotated image -7 degree
        %% rotate Image -1- degree

        im = imrotate(resizedImage, -10,'bilinear', 'crop' );
        %Edge detection of my face from the training database
        [~, threshold] = edge(im, 'sobel');
        BW1 = edge(im,'sobel', threshold * fudgeFactor);

        %extract the hog features of the processed face
        [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
        %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
        feature_matrix(i+(noOfImage*6),:) = featureVector;

        image_class_matrix{i+(noOfImage*6)} = name{1};
        % rotated -7

        %%

    %     im = medfilt2(resizedImage);
    %     %Edge detection of my face from the training database
    %     [~, threshold] = edge(im, 'sobel');
    %     BW1 = edge(im,'sobel', threshold * fudgeFactor);
    % 
    %     %extract the hog features of the processed face
    %     [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
    %     %featureVector = extractLBPFeatures(BW1,'CellSize',[8 8]);
    %     feature_matrix(i+(noOfImage*7),:) = featureVector;
    %     
    %     image_class_matrix{i+(noOfImage*7)} = name{1};
    %     % rotated -7

        ax = subplot(10, 10, i);
        imshow(im, 'Parent', ax);

    end

    %%
    feature_set = array2table(feature_matrix);
    svm_model = fitcecoc(feature_set, image_class_matrix);
    knn_model = fitcknn(feature_set, image_class_matrix,'NumNeighbors',5,'Standardize',1);

    % %Edge detection of my face from the training database
    % [~, threshold] = edge(I, 'sobel');
    % fudgeFactor = .7;
    % BW1 = edge(I,'sobel', threshold * fudgeFactor);
    % 
    % %extract the hog features of the processed face
    % [featureVector,hogVisualization] = extractLBPFeatures(BW1,'CellSize',[8 8]);
    % 
    % %visualize them
    % figure;
    % subplot(1,3,1);
    % imshow(I);
    % subplot(1,3,2);
    % imshow(BW1);
    % subplot(1,3,3);
    % imshow(BW1);
    % hold on;
    % plot (hogVisualization);

     
end  
     
   