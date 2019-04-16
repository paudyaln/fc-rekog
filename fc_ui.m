function varargout = fc_ui(varargin)
% FC_UI MATLAB code for fc_ui.fig
%      FC_UI, by itself, creates a new FC_UI or raises the existing
%      singleton*.
%
%      H = FC_UI returns the handle to a new FC_UI or the handle to
%      the existing singleton*.
%
%      FC_UI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in FC_UI.M with the given input arguments.
%c
%      FC_UI('Property','Value',...) creates a new FC_UI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before fc_ui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to fc_ui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help fc_ui

% Last Modified by GUIDE v2.5 14-Apr-2019 22:49:52

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @fc_ui_OpeningFcn, ...
                   'gui_OutputFcn',  @fc_ui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before fc_ui is made visible.
function fc_ui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to fc_ui (see VARARGIN)

% Choose default command line output for fc_ui
[handles.noOfImage,handles.images,handles.mface, handles.eigen_faces, handles.weights_mat] = face_trainer();
handles.output = hObject;
set(handles.imagepanel,'visible','off');

%set(handles.cameraaxes, 'Units', 'pixels');
resi = get(handles.capturedimage, 'Position');
axes(handles.cameraaxes);
handles.vid = videoinput('winvideo',1,'RGB24_960x540');
src = getselectedsource(handles.vid);
src.Brightness = 200;
disp(src);
get(handles.vid);
hImage= image(zeros(540,960,3), 'Parent', handles.cameraaxes);
preview(handles.vid, hImage);





% Update handles structure
guidata(hObject, handles);

% UIWAIT makes fc_ui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = fc_ui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in capturebutton1.
function capturebutton1_Callback(hObject, eventdata, handles)
% hObject    handle to capturebutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
frame = getsnapshot(handles.vid);
set(handles.capturedimage, 'Units', 'pixels');
resizePos = get(handles.capturedimage, 'Position');
disp(resizePos);
axes(handles.capturedimage);
%disp(size(myImage));
myImage = imresize(rgb2gray(detectFace(frame)),[60 60]);
handles.testface = myImage;
imshow(myImage);
set(handles.capturedimage,'Units','normalized');
set(handles.imagepanel,'visible','on');
guidata( hObject, handles );

%image(frame);


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[handles.mean_square_error] = face_recognizer(handles.noOfImage, handles.eigen_faces, handles.mface, handles.weights_mat, handles.testface);
disp(handles.mean_square_error);
imshow(handles.images{handles.mean_square_error});
set(handles.resultimage,'Units','normalized');
guidata( hObject, handles );





% --- Executes on button press in open_camera.
function open_camera_Callback(hObject, eventdata, handles)
% hObject    handle to open_camera (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Create the face detector object.
faceDetector = vision.CascadeObjectDetector();

% Create the point tracker object.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Create the webcam object.
%cam = webcam();

% Capture one frame to get its size.
videoFrame = getsnapshot(handles.vid);
frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
runLoop = true;
numPts = 0;
frameCount = 0;

while runLoop && frameCount < 400

    % Get the next frame.
    videoFrame = getsnapshot(handles.vid);
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;

    %if numPts < 10
        % Detection mode.
        bbox = faceDetector.step(videoFrameGray);

        if ~isempty(bbox)
            % Find corner points inside the detected region.
            %points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));

            % Re-initialize the point tracker.
            %xyPoints = points.Location;
            %numPts = size(xyPoints,1);
            %release(pointTracker);
            %initialize(pointTracker, xyPoints, videoFrameGray);

            % Save a copy of the points.
            %oldPoints = xyPoints;

            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            bboxPoints = bbox2points(bbox(1, :));

            % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
            % format required by insertShape.
            bboxPolygon = reshape(bboxPoints', 1, []);

            % Display a bounding box around the detected face.
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

            % Display detected corners.
            %videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end

%     else
%         Tracking mode.
%         [xyPoints, isFound] = step(pointTracker, videoFrameGray);
%         visiblePoints = xyPoints(isFound, :);
%         oldInliers = oldPoints(isFound, :);
% 
%         numPts = size(visiblePoints, 1);
% 
%         if numPts >= 10
%             Estimate the geometric transformation between the old points
%             and the new points.
%             [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
%                 oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
% 
%             Apply the transformation to the bounding box.
%             bboxPoints = transformPointsForward(xform, bboxPoints);
% 
%             Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
%             format required by insertShape.
%             bboxPolygon = reshape(bboxPoints', 1, []);
% 
%             Display a bounding box around the face being tracked.
%             videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
% 
%             Display tracked points.
%             videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');
% 
%             Reset the points.
%             oldPoints = visiblePoints;
%             setPoints(pointTracker, oldPoints);
%         end
% 
%     end

    % Display the annotated video frame using the video player object.
    step(videoPlayer, videoFrame);
    %showFrameOnAxis(handles.cameraaxes, videoFrame);
    % Check whether the video player window has been closed.
    %runLoop = isOpen(videoPlayer);
end



Clean up.
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);

