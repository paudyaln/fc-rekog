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
%
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

% Last Modified by GUIDE v2.5 27-Mar-2019 17:43:10

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
disp(resi);
axes(handles.cameraaxes);
handles.vid = videoinput('winvideo');
get(handles.vid);
hImage= image(zeros(480,640,3), 'Parent', handles.cameraaxes);
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
myImage = imresize(rgb2gray(detectFace(frame)),[50 50]);
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



