function [points] = detectFace(inputImg)
    % Create a cascade detector object.
    faceDetector = vision.CascadeObjectDetector();
    
    %Detect Face
    bbox = step(faceDetector, inputImg);
    
    % Draw the returned bounding box around the detected face.
    imgFrame = insertShape(inputImg, 'Rectangle', bbox);
    figure; imshow(imgFrame); title('Detected face');
    
    points = detectMinEigenFeatures(rgb2gray(imgFrame), 'ROI', bbox);
    
     % Display the detected points.
    %figure, imshow(img), hold on, title('Detected features');
    %plot(points);
end

