function [noOfImage,imageSet, mface, eigen_faces, weights_mat] = face_trainer()
    %%% Image training and finding eigenfaces
    imagelist = dir('image_set');
    noOfFile = length(imagelist);
    noOfImage = noOfFile - 2;
    imagelist = imagelist(3:noOfFile);
    disp(noOfImage);
    imageSet = cell(1, noOfImage*2);
    HEset = cell(1, noOfImage);
    %LOGset = cell(1, noOfImage);
    images = cell(1, noOfImage);
    current_dir = replace(pwd, '\', '/');
    image_dir =  '/image_set/';
    disp(current_dir);
    %vectorized_images = zeros(50*50, noOfImage);
    vectorized_images = zeros(3600, noOfImage*2);
    he_images = zeros(3600, noOfImage);
    %%% read each image and store it in the set of matrix
    %%% Vectorize the image matrix
    for i = 1:noOfImage
        filename = strcat(current_dir,image_dir,imagelist(i).name);
        im = rgb2gray(imread(filename));
        images{i} = im;
        resizedImage = imresize(detectFace(im),[60 60]);
        imageSet{i} = resizedImage;
        vectorized_images(:, i) = reshape(double(resizedImage),[], 1);
        %figure;
        %imshow(resizedImage);
        im = histeq(resizedImage);
        HEset{i} = im;
        he_images(:,i) = reshape(double(im),[],1);
        vectorized_images(:, i+19) = reshape(double(im),[],1);
        imageSet{i+19} = im;
    end
    %%% find the mean vector of the vectorize image
    mean_vector = sum(vectorized_images,2) ./ (noOfImage*2);
    mface = mean_vector;
    %average_face = reshape(mean_vector, 60, 60);
    mean_vector = repmat(mean_vector, 1, noOfImage*2);
    substracted_image = vectorized_images - mean_vector;

    %%% Finding the coverance matrix
    coverance_matrix = transpose(substracted_image) * substracted_image;

    %%% find the eigen vector and the eigen values
    [eigen_vec,D] = eigs(coverance_matrix,10);
    [~, desc] = sort(diag(D),'descend');
    eigen_vec = eigen_vec(:, desc);
    %%% Find eigenfaces
    eigen_faces =  substracted_image * eigen_vec;

    % Normalize
    for i = 1 : size(eigen_faces,2)
        eigen_faces(:,i) = eigen_faces(:,i) ./ norm(eigen_faces(:,i),2);    
    end

    %efaces = eigen_faces;
    %eigen_faces = reshape(eigen_faces, 60, 60,  10);

    %Project each training set to the face space and claculate the weights
    weights_mat = zeros(10,noOfImage);
    for i=1:noOfImage*2
       train_face = double(reshape(imageSet{i}, [],1));
       weights_mat(:, i) = transpose(eigen_faces) * (train_face-mface);
    end
end

