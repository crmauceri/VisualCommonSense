function varargout = get_truth(varargin)
% GET_TRUTH MATLAB code for get_truth.fig
%      GET_TRUTH, by itself, creates a new GET_TRUTH or raises the existing
%      singleton*.
%
%      H = GET_TRUTH returns the handle to a new GET_TRUTH or the handle to
%      the existing singleton*.
%
%      GET_TRUTH('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GET_TRUTH.M with the given input arguments.
%
%      GET_TRUTH('Property','Value',...) creates a new GET_TRUTH or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before get_truth_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to get_truth_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help get_truth

% Last Modified by GUIDE v2.5 17-Jun-2015 11:27:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @get_truth_OpeningFcn, ...
                   'gui_OutputFcn',  @get_truth_OutputFcn, ...
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
end

% --- Executes just before get_truth is made visible.
function get_truth_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to get_truth (see VARARGIN)

% Get filepath from varargin
if length(varargin)<4
    display('Arguments required: root directory, search description, query, image_file_id' );
end

handles.root = varargin{1};
search_description = varargin{2};
handles.query = varargin{3};
image_file_id = varargin{4};
handles.errors = 0;
handles.initialize = 1;

addpath('analysis');
addpath('transfer');

h = waitbar(0,'Please wait while we load the nessisary files');

concepts = load(sprintf('%s/concepts/%s_owner_per_concept.mat', handles.root, search_description));
handles.concepts = cellstr(concepts.concepts);
handles.num_concepts = length(handles.concepts);

handles.label_load_str = sprintf('%s/validation/multi_class_%s_%s.mat', handles.root, search_description, image_file_id);
handles.label_save_str = sprintf('%s_processed.mat', handles.label_load_str(1:end-4));

if ~(exist( handles.label_load_str , 'file'))
    msgbox('Target file does not exist');
    get_truth_fig_CloseRequestFcn(hObject, eventdata, handles)
    return
else
    edge_labels = load(handles.label_load_str);
    handles.edge_label_mask = edge_labels.edge_label_mask;
    handles.edge_label = edge_labels.edge_label;
    handles.related = edge_labels.related;
    handles.non_visual = edge_labels.non_visual;
    handles.image_matrix = edge_labels.final_image_matrix;
    handles.image_index = edge_labels.final_image_index;
    handles.relationships = edge_labels.relationships;
    handles.cur_edge = 1;
    handles.dim = edge_labels.dim;
end

if (exist( handles.label_save_str , 'file'))
    edge_labels = load(handles.label_save_str);
    handles.edge_label_mask = edge_labels.edge_label_mask;
    handles.edge_label = edge_labels.edge_label;
    handles.related = edge_labels.related;
    handles.non_visual = edge_labels.non_visual;
    handles.relationships = edge_labels.relationships;
    handles.cur_edge = edge_labels.cur_edge;
end

set(handles.relationship_list, 'String', handles.relationships);

[handles, do_continue] = next_edge(hObject, eventdata, handles);
if do_continue
    % Update handles structure
    guidata(hObject, handles);
end
close(h);

% UIWAIT makes get_truth wait for user response (see UIRESUME)
% uiwait(handles.get_truth_fig);
end

% --- Outputs from this function are returned to the command line.
function varargout = get_truth_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = [];
end

% --- Executes on selection change in relationship_list.
function relationship_list_Callback(hObject, eventdata, handles)
% hObject    handle to relationship_list (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns relationship_list contents as cell array
%        contents{get(hObject,'Value')} returns selected item from relationship_list
end

% --- Executes during object creation, after setting all properties.
function relationship_list_CreateFcn(hObject, eventdata, handles)
% hObject    handle to relationship_list (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end

% --- Executes on button press in enter_relationship.
function enter_relationship_Callback(hObject, eventdata, handles)
% hObject    handle to enter_relationship (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

selected_rel = get(handles.relationship_list, 'Value');
flip_order = get(handles.flip_box, 'Value');
related = get(handles.related_checkbox, 'Value');
non_visual = get(handles.visual_box, 'Value');

handles.related(handles.x, handles.y) = related;
handles.related(handles.y, handles.x) = related;
handles.edge_label_mask(handles.y, handles.x) = 1;
handles.edge_label_mask(handles.x, handles.y) = 1;

if related == 1
    if flip_order == 0
        handles.edge_label(sub2ind(handles.dim, handles.x, handles.y), selected_rel) = 1;
        handles.non_visual(handles.x, handles.y) = non_visual;
    else
        handles.edge_label(sub2ind(handles.dim, handles.y, handles.x), selected_rel) = 1;
        handles.non_visual(handles.y, handles.x) = non_visual;
    end
end

set(handles.flip_box, 'Value', 0);
set(handles.related_checkbox, 'Value', 0);
set(handles.visual_box, 'Value', 0);
set(handles.relationship_list, 'Value', 1);

h = waitbar(0,'Please wait while we load the nessisary files');
[handles, do_continue] = next_edge(hObject, eventdata, handles);
if do_continue
    % Update handles structure
    guidata(hObject, handles);
end

close(h);
end

% --- Executes on button press in save_button.
function save_button_Callback(hObject, eventdata, handles)
% hObject    handle to save_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

edge_label_mask = handles.edge_label_mask;
edge_label = handles.edge_label;
related = handles.related;
non_visual = handles.non_visual;
relationships = handles.relationships;
cur_edge = handles.cur_edge;
save(handles.label_save_str, 'edge_label', 'edge_label_mask', 'related', 'non_visual', 'relationships', 'cur_edge');
msgbox('Labels Saved');
end

%---- Selects the next edge to label.
function [handles, do_continue] = next_edge(hObject, eventdata, handles)
 
if handles.initialize == 1
   handles.x = handles.image_index(1, handles.cur_edge);
   handles.y = handles.image_index(2, handles.cur_edge);
   handles.initialize = 0;
end

if handles.cur_edge<=size(handles.image_index, 2)
   while(handles.edge_label_mask(handles.y, handles.x) == 1)
       handles.cur_edge = handles.cur_edge + 1;
       handles.x = handles.image_index(1, handles.cur_edge);
       handles.y = handles.image_index(2, handles.cur_edge);
   end
   axes(handles.image_axes);
   imshow(handles.image_matrix{handles.cur_edge});    
   edge_str = sprintf('%s, %s', handles.concepts{handles.y}, handles.concepts{handles.x});
   set(handles.title_str, 'String', edge_str); 
   handles.cur_edge = handles.cur_edge + 1;
   do_continue = true;
else
   msgbox('All edges labeled');
   save_button_Callback(hObject, eventdata, handles);
   get_truth_fig_CloseRequestFcn(hObject, eventdata, handles);
   do_continue = false;
end
    
end


% --- Executes on button press in flip_box.
function flip_box_Callback(hObject, eventdata, handles)
% hObject    handle to flip_box (see GCBO)
% eventdata  reserved - to be definedgt in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of flip_box
end


% --- Executes on button press in visual_box.
function visual_box_Callback(hObject, eventdata, handles)
% hObject    handle to visual_box (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of visual_box
end

% --- Executes on button press in related_checkbox.
function related_checkbox_Callback(hObject, eventdata, handles)
% hObject    handle to related_checkbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of related_checkbox
end

% --- Executes when user attempts to close get_truth_fig.
function get_truth_fig_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to get_truth_fig (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
delete(hObject);
end
