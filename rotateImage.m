function [Q] = rotateImage(Q)
    % Dividing the image in two halves for better detection
    % To see why , see this: https://www.mathworks.com/matlabcentral/answers/155126-how-does-the-vision-cascadeobjectdetector-detect-left-and-right-eyes-separately-it-is-constantly-de
    n = fix(size(Q,2)/2);
    lefthalf = Q(:,1:n,:);
    righthalf = Q(:,n+1:end,:);

    RightEyeDetect = vision.CascadeObjectDetector('RightEyeCART');
    LeftEyeDetect = vision.CascadeObjectDetector('LeftEyeCART');
    % vision.CascadeObjectDetector(EyePairBig) is not much efficient in this case
    % because the image is tilted. So, detecting both eyes separately.

    %Bounding Boxes
    BBREye= step(RightEyeDetect,lefthalf); %Right eye is on our left
    BBLEye= step(LeftEyeDetect,righthalf); %Left eye is on our right
    BBLEye(1)=BBLEye(1)+n; %correcting the x position of left eye (caused due to dividing the image in two halves)

    figure
    imshow(imrotate(Q,(180/pi)*atan((BBREye(2)-BBLEye(2))/(BBREye(1)-BBLEye(1)))));
end