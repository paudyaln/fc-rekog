function [inputImg] = detectFace(inputImg)
    % input: Image
    % returns: detected face image
    
    % Create a cascade detector object.
    faceDetector = vision.CascadeObjectDetector();
    
    %Detect Face
    bbox = step(faceDetector, inputImg);
    % Draw the returned bounding box around the detected face.
    %imgFrame = insertShape(inputImg, 'Rectangle', [bbox(1)+20 bbox(2)+20 bbox(3)-40 bbox(4)-40]);
    inputImg = imcrop(inputImg, bbox);
end  
     
   