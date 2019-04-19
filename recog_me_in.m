% GUI runner file
function varargout = recog_me_in(varargin)
% RECOG_ME_IN MATLAB code for recog_me_in.fig
%      RECOG_ME_IN, by itself, creates a new RECOG_ME_IN or raises the existing
%      singleton*.
%
%      H = RECOG_ME_IN returns the handle to a new RECOG_ME_IN or the handle to
%      the existing singleton*.
%
%      RECOG_ME_IN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RECOG_ME_IN.M with the given input arguments.
%
%      RECOG_ME_IN('Property','Value',...) creates a new RECOG_ME_IN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before recog_me_in_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to recog_me_in_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help recog_me_in

% Last Modified by GUIDE v2.5 18-Apr-2019 14:23:37

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @recog_me_in_OpeningFcn, ...
                   'gui_OutputFcn',  @recog_me_in_OutputFcn, ...
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


% --- Executes just before recog_me_in is made visible.
function recog_me_in_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to recog_me_in (see VARARGIN)

% Choose default command line output for recog_me_in
handles.output = hObject;
[handles.knn_model, handles.svm_model] = trainer_feature_extraction();
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes recog_me_in wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = recog_me_in_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in open_cam.
function open_cam_Callback(hObject, eventdata, handles)
% hObject    handle to open_cam (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

r = open_live_recog(handles.knn_model, handles.svm_model);
im = imread(strcat('image_set/',r{1} ,'.jpg'));
set(handles.result_image, 'Units', 'pixels');
%resizePos = get(handles.capturedimage, 'Position');
axes(handles.result_image);
handles.resultface = im;
imshow(im);
set(handles.result_image,'Units','normalized');
set(handles.result_name, 'String',r{1});
disp(r);
guidata(hObject, handles);
