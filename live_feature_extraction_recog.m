faceDetector = vision.CascadeObjectDetector();

faceDetector.MinSize = [175, 175];

pointTracker = vision.PointTracker('MaxBidirectionalError',2);


%create the webcam object 
cam = webcam(1);
%cam.Resolution = '640*480';

videoFrame = snapshot(cam);
frameSize = size(videoFrame);

videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

runLoop = true;
numPts = 0;
svm = trainer_feature_extraction();

while runLoop
    videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    
    if numPts < 50
        bbox = faceDetector.step(videoFrameGray);
    
    if ~isempty(bbox)
        points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1,:));
        
        xyPoints = points.Location;
        numPts = size(xyPoints,1);
        release(pointTracker);
        initialize(pointTracker, xyPoints, videoFrameGray);
        
        
        oldPoints = xyPoints;
        
        % Convert the rectangle represented as [x, y, w, h] into an
         % M-by-2 matrix of [x,y] coordinates of the four corners. This
         % is needed to be able to transform the bounding box to display
         % the orientation of the face.
         bboxPoints = bbox2points(bbox(1, :));
        bboxPolygon = reshape(bboxPoints', 1, []);
        
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWIdth', 3);
        
        videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
    end
    else
        % Tracking mode.
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints,1);
        
        if numPts >= 50
           [xform, oldInliers, visiblePoints] = estimateGeometricTransform(oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4); 
            
        % Apply the transformation to the bounding box.
         bboxPoints = transformPointsForward(xform, bboxPoints);
         % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
         % format required by insertShape.
         bboxPolygon = reshape(bboxPoints', 1, []);
         % Display a bounding box around the face being tracked.
         videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
         % Display tracked points.
         videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
 
         % Reset the points.
         oldPoints = visiblePoints;
         setPoints(pointTracker, oldPoints);
        end
    end
    
    inputImg = imresize(imcrop(videoFrameGray, bbox),[100 100]);
    %Edge detection of my face from the training database
    [~, threshold] = edge(inputImg, 'sobel');
    fudgeFactor = .7;
    BW1 = edge(inputImg,'sobel', threshold * fudgeFactor);
    [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
    [predicted_test, score_test, cost_test] = predict(svm, featureVector);

    disp(predicted_test);
    
    % Display the annotated video frame using the video player object.
     step(videoPlayer, videoFrame);
     % Check whether the video player window has been closed.
     runLoop = isOpen(videoPlayer);
end
    
    
    % Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);