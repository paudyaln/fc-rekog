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

% Last Modified by GUIDE v2.5 25-Mar-2019 22:00:08

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
handles.output = hObject;
axes(handles.axes1);
vid = videoinput('winvideo');
hImage= image(zeros(600,700,3), 'Parent', handles.axes1);

preview(vid, hImage);

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
