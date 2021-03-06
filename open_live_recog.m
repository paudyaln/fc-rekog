function [result] = open_live_recog(knn, svm)
    %knn    KNN face recognition Model
    %svm    SVM based face recognition Model
    % Open up the camera, extract frame and recognize faces, 
    
    
    %initialize haar cascader
    faceDetector = vision.CascadeObjectDetector();
    faceDetector.MinSize = [175, 175];

    pointTracker = vision.PointTracker('MaxBidirectionalError',2);


    %create the webcam object 
    cam = webcam(1);
    %cam.Resolution = '640*480';

    videoFrame = snapshot(cam);
    frameSize = size(videoFrame);

    videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

    A = [];
    runLoop = true;
    counter = 0;
    numPts = 0;
    while runLoop && counter < 100
        counter = counter + 1;
        videoFrame = snapshot(cam);
        videoFrameGray = rgb2gray(videoFrame);

        if numPts < 30
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

            if numPts >= 30
               [xform, oldInliers, visiblePoints] = estimateGeometricTransform(oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4); 

               inputImg = imresize(imcrop(videoFrameGray, bbox),[100 100]);
            %Edge detection of my face from the training database
            [~, threshold] = edge(inputImg, 'sobel');
            fudgeFactor = .7;
            BW1 = edge(inputImg,'sobel', threshold * fudgeFactor);
            [featureVector,hogVisualization] = extractHOGFeatures(BW1,'CellSize',[8 8]);
            [predicted_test, score_test, cost_test] = predict(svm, featureVector);
            [predictedLabels_knn, score_knn, cost_knn] = predict(knn, featureVector);

            %disp(predicted_test);
            A = [A,predicted_test];

            % Apply the transformation to the bounding box.
             bboxPoints = transformPointsForward(xform, bboxPoints);
             % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
             % format required by insertShape.
             bboxPolygon = reshape(bboxPoints', 1, []);
             % Display a bounding box around the face being tracked.
             videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
             %Insert name to the predicted image frame
             if(strcmp(predicted_test{1}, predictedLabels_knn{1}) == 1)
                videoFrame = insertText(videoFrame,  [round(bboxPolygon(1)) round(bboxPolygon(2))],predicted_test{1},'FontSize',18,'BoxColor',...
                    'red','BoxOpacity',0.4,'TextColor','white');
             end
             % Display tracked points.
             videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

             % Reset the points.
             oldPoints = visiblePoints;
             setPoints(pointTracker, oldPoints);
            end
        end



        % Display the annotated video frame using the video player object.
         step(videoPlayer, videoFrame);
         % Check whether the video player window has been closed.
         runLoop = isOpen(videoPlayer);
    end
    U = unique(A);
    %disp(U);
    n = zeros(length(U), 1);
    for iU = 1:length(U)
      n(iU) = length(find(strcmp(U{iU}, A)));
    end
    [~, itemp] = max(n);
    result = U(itemp);
    disp('You are');
    disp(result);

    % Clean up.
    clear cam;
    release(videoPlayer);
    release(pointTracker);
    release(faceDetector);
     
end  
     