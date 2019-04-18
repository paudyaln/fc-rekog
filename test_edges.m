close all;
im = imresize(rgb2gray(detectFace(imread('Nischal_Paudyal-3.jpg'))),[100 100]);
a = imadjust(im, [0.3 0.7]);
im = medfilt2(im);
[~, threshold] = edge(im, 'sobel');
fudgeFactor = .5;
BW1 = edge(im,'sobel', threshold * fudgeFactor);
figure;
imshow(BW1);


im1 = imresize(rgb2gray(detectFace(imread('nischal_test.jpg'))),[100 100]);
b = imadjust(im1, [0.3 0.7]);
im1 = medfilt2(im1);
[~, threshold1] = edge(im1, 'sobel');
BW2 = edge(im1,'sobel', threshold1 * fudgeFactor);
figure;
imshow(BW2);