msg = 'Hello, World!'
print(msg)


# TODO
# Add sliding window overlap field
# Add color proximity to seizure
# Add day/night colors
# Add stim color


import PySimpleGUI as sg
import os.path
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import math
import pandas as pd
import tools.utils_functions as utils_functions
import tools.latent_plotting as latent_plotting
import pickle
import shutil
import random

def draw_figure(canvas, figure):
   figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
   figure_canvas_agg.draw()
   figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
   return figure_canvas_agg

def plot_latent_on_canvas(window,curr_canvas_agg_tab1, curr_canvas_agg_tab2, latent_data, win_style, samp_freq, start_datetime,stop_datetime,abs_start_datetime, 
                          abs_stop_datetime, win_sec, stride_sec, preictal_dur, 
                          postictal_dur, seiz_start_dt, seiz_stop_dt, absolute_latent, max_latent, auto_scale_plot, interictal_contour,
                          tab2_pickle_file, S_PLOT=1):
    gs1 = gridspec.GridSpec(1, 1)
    fig1 = plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(gs1[0, 0])

    latent_plotting.plot_latent(ax=ax1, latent_data=latent_data, 
                        win_style=win_style,
                        samp_freq=samp_freq, 
                        start_datetime=start_datetime,
                        stop_datetime=stop_datetime,
                        abs_start_datetime=abs_start_datetime, 
                        abs_stop_datetime=abs_stop_datetime, 
                        win_sec=win_sec, 
                        stride_sec=stride_sec,
                        seiz_start_dt=seiz_start_dt, 
                        seiz_stop_dt=seiz_stop_dt, 
                        preictal_dur=preictal_dur,
                        postictal_dur=postictal_dur,
                        absolute_latent=absolute_latent,
                        max_latent=max_latent,
                        auto_scale_plot=auto_scale_plot,
                        interictal_contour=interictal_contour,
                        s_plot=S_PLOT)
    
    # TAB 2
    gs2 = gridspec.GridSpec(1, 1)
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = plt.subplot(gs2[0, 0])

    latent_plotting.plot_latent(ax=ax2, latent_data=latent_data, 
                        win_style=win_style,
                        samp_freq=samp_freq, 
                        start_datetime=start_datetime,
                        stop_datetime=stop_datetime,
                        abs_start_datetime=abs_start_datetime, 
                        abs_stop_datetime=abs_stop_datetime, 
                        win_sec=win_sec, 
                        stride_sec=stride_sec,
                        seiz_start_dt=seiz_start_dt, 
                        seiz_stop_dt=seiz_stop_dt, 
                        preictal_dur=preictal_dur,
                        postictal_dur=postictal_dur,
                        absolute_latent=absolute_latent,
                        max_latent=max_latent,
                        tab2_lighten=True,
                        alpha=1.0,
                        auto_scale_plot=auto_scale_plot,
                        interictal_contour=interictal_contour,
                        s_plot=S_PLOT) 

    
    # Delete current TAB1 plot if present
    if curr_canvas_agg_tab1 != []:
        if 'curr_canvas_agg_tab1' in locals(): curr_canvas_agg_tab1.get_tk_widget().forget()
    curr_canvas_agg_tab1 = draw_figure(window['-CANVAS TAB1-'].TKCanvas, fig1)

        # Delete current TAB2 plot if present
    if curr_canvas_agg_tab2 != []:
        if 'curr_canvas_agg_tab2' in locals(): curr_canvas_agg_tab2.get_tk_widget().forget()
    curr_canvas_agg_tab2 = draw_figure(window['-CANVAS TAB2-'].TKCanvas, fig2)
    

    # Pickle TAB2 figure to pull up later for SPES plotting
    output = open(tab2_pickle_file, 'wb')
    pickle.dump(curr_canvas_agg_tab2.figure, output)
    output.close()


    return curr_canvas_agg_tab1, curr_canvas_agg_tab2

def add_spes_to_plot_tab2(
        window, 
        curr_canvas_agg_tab2,
        latent_data,
        win_style,
        samp_freq,
        start_datetime, 
        stop_datetime, 
        abs_start_datetime,
        abs_stop_datetime,
        win_sec,
        stride_sec,
        seiz_start_dt,
        seiz_stop_dt,
        preictal_dur,
        postictal_dur,
        absolute_latent,
        max_latent,
        auto_scale_plot,
        interictal_contour,
        tab2_pickle_file):


    # Pull the original TAB2 plot out of pickle tmp file
    pkl_file = open(tab2_pickle_file, 'rb')
    curr_canvas_agg_tab2.figure = pickle.load(pkl_file)
    pkl_file.close()

    fig_spes = curr_canvas_agg_tab2.figure
    ax_spes = curr_canvas_agg_tab2.figure.axes[0]

    latent_plotting.plot_latent(ax=ax_spes, latent_data=latent_data, 
                        win_style=win_style,
                        samp_freq=samp_freq, 
                        start_datetime=start_datetime,
                        stop_datetime=stop_datetime,
                        abs_start_datetime=abs_start_datetime, 
                        abs_stop_datetime=abs_stop_datetime, 
                        win_sec=win_sec, 
                        stride_sec=stride_sec,
                        seiz_start_dt=seiz_start_dt, 
                        seiz_stop_dt=seiz_stop_dt, 
                        preictal_dur=preictal_dur,
                        postictal_dur=postictal_dur,
                        absolute_latent=absolute_latent,
                        max_latent=max_latent,
                        alpha=1.0,
                        SPES_colorbar=True,
                        auto_scale_plot=auto_scale_plot,
                        interictal_contour=interictal_contour)

    # Delete current TAB2 plot if present
    if curr_canvas_agg_tab2 != []:
        if 'curr_canvas_agg_tab2' in locals(): curr_canvas_agg_tab2.get_tk_widget().forget()
    curr_canvas_agg_tab2 = draw_figure(window['-CANVAS TAB2-'].TKCanvas, fig_spes)
    
    # Return the modified object
    return curr_canvas_agg_tab2

def compile_seiz_list(start_dt, stop_dt, seiz_types):

    start_strs = [d.strftime("%m/%d/%Y-%H:%M:%S") for d in start_dt]
    stop_strs = [d.strftime("%m/%d/%Y-%H:%M:%S") for d in stop_dt]
    list_str = [start_strs[i] + ' to ' + stop_strs[i] + " [" + seiz_types[i] + "]" for i in range(0,len(seiz_types))]

    return list_str


def get_random_interictal_epoch(
        abs_start_datetime, abs_stop_datetime, pat_seiz_start_datetimes, pat_seiz_stop_datetimes,
        random_epoch_seconds, start_EMU_buffer_hours, seizure_buffer_hours):

    # Get how many seconds in EMU stay
    total_EMU_seconds = int((abs_stop_datetime[0] - abs_start_datetime[0]).total_seconds())
    
    # Find a random second in EMU stay that allows for a window to be created away from seizures
    rand_start_dt = []
    rand_stop_dt = []
    epoch_found = False
    while not epoch_found:
        rand_sec = random.randrange(0, total_EMU_seconds, 1)
        rand_start_dt = abs_start_datetime[0] + datetime.timedelta(seconds=rand_sec)
        rand_stop_dt = rand_start_dt + datetime.timedelta(seconds=random_epoch_seconds)

        # Check if this window meets requirements
        if rand_start_dt < (abs_start_datetime[0] + datetime.timedelta(hours=start_EMU_buffer_hours)):
            continue
        
        skip = False
        for i in range(0, len(pat_seiz_start_datetimes)):
            curr_seiz_start_buffered = pat_seiz_start_datetimes[i] - datetime.timedelta(hours=seizure_buffer_hours)
            curr_seiz_stop_buffered = pat_seiz_stop_datetimes[i] + datetime.timedelta(hours=seizure_buffer_hours)

            if ((rand_start_dt > curr_seiz_start_buffered) & (rand_start_dt < curr_seiz_stop_buffered)) | ((rand_stop_dt > curr_seiz_start_buffered) & (rand_stop_dt < curr_seiz_stop_buffered)) :
                skip = True
                break

        if skip == False:
            epoch_found = True

    return rand_start_dt, rand_stop_dt


SAMP_FREQ = 512 # Hz, assumed
DEFAULT_WIN = 10  # seconds
DEFAULT_STRIDE = 1  # seconds
DEFAULT_PREICTAL_DUR = 120
DEFAULT_POSTICTAL_DUR = 120
ABSOLUTE_LATENT = False
MAX_LATENT = False
DEFAULT_WIN_STYLE = 'end'
DEFAULT_FBTC=True
DEFAULT_FIAS=True
DEFAULT_FAStoFIAS=True
DEFAULT_FAS=True
DEFAULT_FOCAL_UNKNOWN=True
DEFAULT_UNKNOWN=True
DEFAULT_SUBCLINICAL=False
DEFAULT_NONELECTRO=False
DEFAULT_INTERICTAL_CONTOUR=False

DEFAULT_AUTO_SCALE_PLOT = True
DEFAULT_PERIICTAL_BUFFER_SEC = 1200

DEFAULT_START_EMU_BUFFER_HOURS = 4
DEFAULT_SEIZURE_BUFFER_HOURS = 2

DEFAULT_RANDOM_EPOCH_SECONDS = 60

EXAMPLE_LATENT_FILE = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/Same_Scale_For_All_Channels/HistEqualScale/data_normalized_to_first_seizure_centered_24_hours/DurStr_512s480s_epoched_datasets/Epat27/trained_models/dataset_train100.0_val0.0_test0.0/A/Epat27_master_latent_inference_on_all_epochs_withSPES_trainedepoch59_02182020_17092099_to_02202020_16471496.pkl'

possible_win_styles = ['end', 'centered']

# All time data file from preprocessing
DEFAULT_ATD_DIR = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data'
DEFAULT_ATD_FILE = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/all_time_data_01092023_112957.csv'
DEFAULT_SPES_DIR = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/data/SPES_Natus_Timestamps'
atd_df = pd.read_csv(DEFAULT_ATD_FILE, sep='\t')
all_patIDs = atd_df['Pat ID'].unique()
all_patID_list = [str(x) for x in all_patIDs]  

DEFAULT_BROWSING_LATENT_DIR = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results'

DEFAULT_SPES_STRIDE = 0.01

# First the window layout in 2 columns
ch_names_prev = -1

tab1_file_list_column = [
    [
        sg.In(size=(30, 1), enable_events=True, key="-ATD FILE-"),
        sg.FileBrowse(initial_folder=DEFAULT_ATD_DIR),
        sg.Text("All Patient Time Data (.csv)"),
    ],
    [
        sg.Combo(all_patID_list, font=('Arial Bold', 9),  size=(30, 1), expand_x=False, enable_events=True,  readonly=False, key='-PAT ID-'),
        sg.Text("Patient ID")
    ],
    [
        sg.In(size=(30, 1), enable_events=True, key="-LATENT FILE-"),
        sg.FileBrowse(initial_folder=DEFAULT_BROWSING_LATENT_DIR),
        sg.Text("Master Latent File (.pkl)"),
    ],
    [
        sg.Combo('', font=('Arial Bold', 9),  size=(60, 1), expand_x=False, enable_events=True,  readonly=False, key='-SEIZURE SELECTION-'),
        sg.Text("Jump To Seizure")
    ],
    [
        sg.Input('', enable_events=True, size=(30, 5), key="-START TIMESTAMP INPUT-"),
        sg.Text("Start Timestamp"),
        sg.Button('Set to EMU Start', key="-RESET STARTTIME-")
    ],
    [
        sg.Input('', enable_events=True, size=(30, 5), key="-STOP TIMESTAMP INPUT-"),
        sg.Text("Stop Timestamp"),
        sg.Button('Set to EMU End', key="-RESET STOPTIME-")
    ],
    [
        sg.Input('', enable_events=True, size=(30, 5), key="-WINDOW DUR-"),
        sg.Text("Window Duration to Mean, sec")
    ],
    [
        sg.Input('', enable_events=True, size=(30, 5), key="-STRIDE-"),
        sg.Text("Stride, sec")
    ],
    # [
    #     sg.Combo(possible_win_styles, size=(28, 5), font=('Arial Bold', 9),  expand_x=False, enable_events=True,  readonly=True, key='-WIN STYLE-')
    # ],
    [
        sg.Input('', enable_events=True, size=(30, 5), key="-PREICTAL DUR-"),
        sg.Text('Pre-Ictal Coloring Duration, sec')
    ],
    [
        sg.Input('', enable_events=True, size=(30, 5), key="-POSTICTAL DUR-"),
        sg.Text('Post-Ictal Coloring Duration, sec')
    ],
    [
        sg.Text("Latent Space Manipulations:")
    ],
    [
        sg.Checkbox('Abs(Latent)', enable_events=True, key="-ABS LATENT-")
    ],
    [
        sg.Checkbox('Max(Latent)', enable_events=True, key="-MAX LATENT-")
    ],
    [
        sg.Text("Plot Visualization"),
    ],
    [
        sg.Checkbox('Auto Scale Plot', enable_events=True, key="-AUTO SCALE-")
    ],
    [
        sg.Checkbox('Plot Interictal Contour Only', enable_events=True, key="-INTERICTAL CONTOUR-")
    ],
    [
        sg.Text("Seizure Types to Highlight:"),
    ],
    [
        sg.Checkbox('FBTC', enable_events=True, key="-FBTC-")
    ],
    [
        sg.Checkbox('FIAS', enable_events=True, key="-FIAS-")
    ],
    [
        sg.Checkbox('FAS-to-FIAS', enable_events=True, key="-FAStoFIAS-")
    ],
    [
        sg.Checkbox('FAS', enable_events=True, key="-FAS-")
    ],
    [
        sg.Checkbox('Focal, unknown awareness', enable_events=True, key="-FOCAL UNKNOWN-")
    ],
    [
        sg.Checkbox('Unknown Type', enable_events=True, key="-UNKNOWN TYPE-")
    ],
    [
        sg.Checkbox('Subclinical', enable_events=True, key="-SUBCLINICAL-")
    ],
    [
        sg.Checkbox('Non-Electrographic', enable_events=True, key="-NONELECTRO TYPE-")
    ],
    [
        sg.Button('Update Plot', key="-UPDATE BUTTON-")
    ],
    [
        sg.Button('Reset Plot', key="-RESET BUTTON-")
    ],
]

tab2_file_list_column = [
    [
        sg.In(size=(30, 1), enable_events=True,  key="-SPES NOTE FILE-"),
        sg.FileBrowse(initial_folder=DEFAULT_SPES_DIR),
        sg.Text('Load SPES Session'),
    ],
    [
        sg.Listbox(values=[], enable_events=True, size=(60, 39), key='-STIM PAIR LIST-')
    ],
    # [
    #     sg.Input(str(DEFAULT_SPES_WIN), enable_events=True, size=(30, 5), key="-SPES WINDOW DUR-"),
    #     sg.Text("Window Duration to Mean, sec")
    # ],
    [
        sg.Text("Window Duration Locked to 'Latent Plotting' Tab")
    ],
    [
        sg.Input(str(DEFAULT_SPES_STRIDE), enable_events=True, size=(30, 5), key="-SPES STRIDE-"),
        sg.Text("Stride, sec")
    ],
    [
        sg.Text("------------")
    ],
    [
        sg.Button('Plot Random Interictal Epoch', key="-RANDOM INTERICTAL-")
    ],
    [
        sg.Input(str(DEFAULT_RANDOM_EPOCH_SECONDS), enable_events=True, size=(15, 5), key="-RANDOM INTERICTAL DURATION-"),
        sg.Text("Random interictal duration, sec")
    ],
]

# For now will only show the name of the file that was chosen
tab1_image_viewer_column = [
    # [sg.Text("...", key="-INSTRUCTIONS-")],
    # [sg.Text(size=(100, 1), key="-TOUT-")],
    [sg.Canvas(key="-CANVAS TAB1-")],
]

tab2_image_viewer_column = [
    # [sg.Text("...", key="-INSTRUCTIONS-")],
    # [sg.Text(size=(100, 1), key="-TOUT-")],
    [sg.Canvas(key="-CANVAS TAB2-")],
]

# ----- Full layout -----
tab1_layout = [[sg.Column(tab1_file_list_column), 
                sg.VSeperator(), 
                sg.Column(tab1_image_viewer_column)]]

tab2_layout = [[sg.Column(tab2_file_list_column), 
                sg.VSeperator(), 
                sg.Column(tab2_image_viewer_column)]]

layout = [[sg.TabGroup([[
                    sg.Tab('Latent Plotting', tab1_layout),
                    sg.Tab('SPES', tab2_layout),
                    ]]), 
                    # sg.Button('Close')
          ]]  

# Initialize the GUI
window = sg.Window("Latent Space Explorer (LSE)", layout)
curr_canvas_agg_tab1 = []
curr_canvas_agg_tab2 = []
pat_seiz_start_datetimes = []
pat_seiz_stop_datetimes = []
seiz_types = []
seizures_selection_list = []
abs_start_datetime = []
abs_stop_datetime = []

spes_master_EDF_creation_datetime = []
spes_stim_pair_names = []
spes_start_datetimes = []
spes_stop_datetimes = []
printable_all_stim_pairs_dt = []

# For the TAB2 pickle
tmp_dir = os.getcwd() + '/tmp_files_LSE'
if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
os.makedirs(tmp_dir)
tab2_pickle_file = tmp_dir + '/tab2_fig.pkl'

# Run the Event Loop
while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "-ATD FILE-":
        new_file = values['-ATD FILE-']
        atd_df = pd.read_csv(new_file, sep='\t')
        all_patIDs = atd_df['Pat ID'].unique()
        all_patID_list = [str(x) for x in all_patIDs]  
        window['-PAT ID-'].update(values=all_patID_list)

    if event == "-PAT ID-":
        pat_id = values["-PAT ID-"]

    if event == "-SPES NOTE FILE-":
        spes_file = values['-SPES NOTE FILE-']

        spes_stim_pair_names, spes_start_datetimes, spes_stop_datetimes = utils_functions.digest_SPES_notes(spes_file)

        # Craft the printable list (Leave in chronological time for now)
        printable_all_stim_pairs_dt = [spes_stim_pair_names[i] + ', Start: ' + spes_start_datetimes[i].strftime('%d/%m/%Y %H:%M:%S') + ' Stop: ' + spes_stop_datetimes[i].strftime('%d/%m/%Y %H:%M:%S') for i in range(0,len(spes_stim_pair_names))]
        window['-STIM PAIR LIST-'].update(values=printable_all_stim_pairs_dt)

    elif event == "-STIM PAIR LIST-": 
        
        # Plotting code 
        selected_str = values["-STIM PAIR LIST-"][0]
        idx = [idx for idx, s in enumerate(printable_all_stim_pairs_dt) if selected_str in s][0]
        spes_plot_start_datetime = spes_start_datetimes[idx]
        spes_plot_stop_datetime = spes_stop_datetimes[idx]
        auto_scale_plot = values['-AUTO SCALE-']
        interictal_contour = values['-INTERICTAL CONTOUR-']

        new_win_sec = float(values['-WINDOW DUR-'])
        new_stride = float(values['-SPES STRIDE-'])

        # Buffer the start time to avoid averaging previous SPES session
        spes_plot_start_datetime_buffered = spes_plot_start_datetime + datetime.timedelta(seconds=new_win_sec)

        curr_canvas_agg_tab2 = add_spes_to_plot_tab2(window, 
                    curr_canvas_agg_tab2=curr_canvas_agg_tab2,
                    latent_data=latent_data,
                    win_style=DEFAULT_WIN_STYLE,
                    samp_freq=SAMP_FREQ,
                    start_datetime=[spes_plot_start_datetime_buffered], 
                    stop_datetime=[spes_plot_stop_datetime], 
                    abs_start_datetime=abs_start_datetime,
                    abs_stop_datetime=abs_stop_datetime,
                    win_sec=new_win_sec,
                    stride_sec=new_stride,
                    seiz_start_dt=pat_seiz_start_datetimes,
                    seiz_stop_dt=pat_seiz_stop_datetimes,
                    preictal_dur=0,
                    postictal_dur=0,
                    absolute_latent=ABSOLUTE_LATENT,
                    max_latent=MAX_LATENT,
                    auto_scale_plot=auto_scale_plot,
                    interictal_contour=interictal_contour,
                    tab2_pickle_file=tab2_pickle_file)

    elif event == "-RESET STARTTIME-":
        window['-START TIMESTAMP INPUT-'].update(abs_start_datetime[0].strftime("%m/%d/%Y-%H:%M:%S"))

    elif event == "-RESET STOPTIME-":
        window['-STOP TIMESTAMP INPUT-'].update(abs_stop_datetime[0].strftime("%m/%d/%Y-%H:%M:%S"))

    elif event == "-RANDOM INTERICTAL-":

        new_win_sec = float(values['-WINDOW DUR-'])
        new_stride = float(values['-SPES STRIDE-'])

        fbtc_bool=values['-FBTC-']
        fias_bool=values['-FIAS-']
        fas2fias_bool=values['-FAStoFIAS-']
        fas_bool=values['-FAS-']
        focal_unknown_bool=values['-FOCAL UNKNOWN-']
        subclinical_bool=values['-SUBCLINICAL-']
        unknown_bool=values['-UNKNOWN TYPE-']
        non_electro_bool=values['-NONELECTRO TYPE-']
        interictal_contour=values['-INTERICTAL CONTOUR-']

        new_random_epoch_seconds = float(values['-RANDOM INTERICTAL DURATION-'])

        pat_seiz_start_datetimes, pat_seiz_stop_datetimes, seiz_types = utils_functions.get_pat_seiz_datetimes(
            pat_id=pat_id,
            FBTC_bool=fbtc_bool,
            FIAS_bool=fias_bool,
            FAS_to_FIAS_bool=fas2fias_bool,
            FAS_bool=fas_bool,
            focal_unknown_bool=focal_unknown_bool,
            subclinical_bool=subclinical_bool,
            unknown_bool=unknown_bool,
            non_electro_bool=non_electro_bool)

        # Get a random interictal start and stop to use for plotting
        interictal_start_datetime, interictal_stop_datetime = get_random_interictal_epoch(
            abs_start_datetime, abs_stop_datetime, pat_seiz_start_datetimes, pat_seiz_stop_datetimes, 
            new_random_epoch_seconds, DEFAULT_START_EMU_BUFFER_HOURS, DEFAULT_SEIZURE_BUFFER_HOURS)
       
        auto_scale_plot = values['-AUTO SCALE-']

        # Buffer the start time to avoid averaging previous SPES session
        interictal_start_datetime_buffered = interictal_start_datetime + datetime.timedelta(seconds=new_win_sec)

        curr_canvas_agg_tab2 = add_spes_to_plot_tab2(window, 
                    curr_canvas_agg_tab2=curr_canvas_agg_tab2,
                    latent_data=latent_data,
                    win_style=DEFAULT_WIN_STYLE,
                    samp_freq=SAMP_FREQ,
                    start_datetime=[interictal_start_datetime_buffered], 
                    stop_datetime=[interictal_stop_datetime], 
                    abs_start_datetime=abs_start_datetime,
                    abs_stop_datetime=abs_stop_datetime,
                    win_sec=new_win_sec,
                    stride_sec=new_stride,
                    seiz_start_dt=pat_seiz_start_datetimes,
                    seiz_stop_dt=pat_seiz_stop_datetimes,
                    preictal_dur=0,
                    postictal_dur=0,
                    absolute_latent=ABSOLUTE_LATENT,
                    max_latent=MAX_LATENT,
                    auto_scale_plot=auto_scale_plot,
                    interictal_contour=interictal_contour,
                    tab2_pickle_file=tab2_pickle_file)

    elif event == "-SEIZURE SELECTION-":
       
        if values['-SEIZURE SELECTION-'] in seizures_selection_list:
            seiz_index = seizures_selection_list.index(values['-SEIZURE SELECTION-'])
        
        seiz_selection_start_dt = pat_seiz_start_datetimes[seiz_index]
        seiz_selection_stop_dt = pat_seiz_stop_datetimes[seiz_index]

        new_win_sec = float(values['-WINDOW DUR-'])
        new_stride = float(values['-STRIDE-'])
        new_preictal = float(values['-PREICTAL DUR-'])
        new_postictal = float(values['-POSTICTAL DUR-'])
        abs_val = values['-ABS LATENT-']
        max_val = values['-MAX LATENT-']
        auto_scale_plot = values['-AUTO SCALE-']
        interictal_contour = values['-INTERICTAL CONTOUR-']

        start_buffered_dt = seiz_selection_start_dt - datetime.timedelta(seconds=new_preictal + DEFAULT_PERIICTAL_BUFFER_SEC)
        stop_buffered_dt = seiz_selection_stop_dt + datetime.timedelta(seconds=new_postictal + DEFAULT_PERIICTAL_BUFFER_SEC)

        window['-START TIMESTAMP INPUT-'].update(start_buffered_dt.strftime("%m/%d/%Y-%H:%M:%S"))
        window['-STOP TIMESTAMP INPUT-'].update(stop_buffered_dt.strftime("%m/%d/%Y-%H:%M:%S"))

        # Currently coded up to set all seizure types to True
        window["-FBTC-"].update(value=True)
        window["-FIAS-"].update(value=True)
        window["-FAStoFIAS-"].update(value=True)
        window["-FAS-"].update(value=True)
        window['-FOCAL UNKNOWN-'].update(value=True)
        window["-SUBCLINICAL-"].update(value=True)
        window["-UNKNOWN TYPE-"].update(value=True)
        window["-NONELECTRO TYPE-"].update(value=DEFAULT_NONELECTRO)


        curr_canvas_agg_tab1, curr_canvas_agg_tab2 = plot_latent_on_canvas(window, 
                    curr_canvas_agg_tab1=curr_canvas_agg_tab1,
                    curr_canvas_agg_tab2=curr_canvas_agg_tab2,
                    latent_data=latent_data,
                    win_style=DEFAULT_WIN_STYLE,
                    samp_freq=SAMP_FREQ,
                    start_datetime=start_buffered_dt, 
                    stop_datetime=stop_buffered_dt, 
                    abs_start_datetime=abs_start_datetime,
                    abs_stop_datetime=abs_stop_datetime,
                    win_sec=new_win_sec,
                    stride_sec=new_stride,
                    seiz_start_dt=pat_seiz_start_datetimes,
                    seiz_stop_dt=pat_seiz_stop_datetimes,
                    preictal_dur=new_preictal,
                    postictal_dur=new_postictal,
                    absolute_latent=abs_val,
                    max_latent=max_val,
                    auto_scale_plot=auto_scale_plot,
                    interictal_contour=interictal_contour,
                    tab2_pickle_file=tab2_pickle_file)

    elif event == "-LATENT FILE-":
        latent_file = values["-LATENT FILE-"]

        # Load the latent pickle file
        with open(latent_file, "rb") as f: latent_data = pickle.load(f)
        if len(latent_data.shape) == 2: 
            print("Expanding 2D to dummy 3D to work with latent plottinig")
            latent_data = np.expand_dims(latent_data,0)
            
        abs_start_datetime, abs_stop_datetime = utils_functions.get_start_stop_from_latent_path(latent_file)
        abs_start_datetime = [abs_start_datetime]
        abs_stop_datetime = [abs_stop_datetime]
        
        
        window['-START TIMESTAMP INPUT-'].update(abs_start_datetime[0].strftime("%m/%d/%Y-%H:%M:%S"))
        window['-STOP TIMESTAMP INPUT-'].update(abs_stop_datetime[0].strftime("%m/%d/%Y-%H:%M:%S"))
        window['-WINDOW DUR-'].update(str(DEFAULT_WIN))
        window['-STRIDE-'].update(str(DEFAULT_STRIDE))
        window["-ABS LATENT-"].update(value=ABSOLUTE_LATENT)
        window["-MAX LATENT-"].update(value=MAX_LATENT)
        window["-AUTO SCALE-"].update(value=DEFAULT_AUTO_SCALE_PLOT)
        # window["-WIN STYLE-"].update(value=DEFAULT_WIN_STYLE)
        window['-PREICTAL DUR-'].update(str(DEFAULT_PREICTAL_DUR))
        window['-POSTICTAL DUR-'].update(str(DEFAULT_POSTICTAL_DUR))
        window["-FBTC-"].update(value=DEFAULT_FBTC)
        window["-FIAS-"].update(value=DEFAULT_FIAS)
        window["-FAStoFIAS-"].update(value=DEFAULT_FAStoFIAS)
        window["-FAS-"].update(value=DEFAULT_FAS)
        window['-FOCAL UNKNOWN-'].update(value=DEFAULT_FOCAL_UNKNOWN)
        window["-UNKNOWN TYPE-"].update(value=DEFAULT_UNKNOWN)
        window["-SUBCLINICAL-"].update(value=DEFAULT_SUBCLINICAL)
        window["-NONELECTRO TYPE-"].update(value=DEFAULT_NONELECTRO) 
        window["-INTERICTAL CONTOUR-"].update(value=DEFAULT_INTERICTAL_CONTOUR)

        pat_seiz_start_datetimes, pat_seiz_stop_datetimes, seiz_types = utils_functions.get_pat_seiz_datetimes(
            pat_id=pat_id,
            FBTC_bool=DEFAULT_FBTC,
            FIAS_bool=DEFAULT_FIAS,
            FAS_to_FIAS_bool=DEFAULT_FAStoFIAS,
            FAS_bool=DEFAULT_FAS,
            focal_unknown_bool=DEFAULT_FOCAL_UNKNOWN,
            subclinical_bool=DEFAULT_SUBCLINICAL,
            unknown_bool=DEFAULT_UNKNOWN,
            non_electro_bool=DEFAULT_NONELECTRO
            )
        

        seizures_selection_list = compile_seiz_list(pat_seiz_start_datetimes,pat_seiz_stop_datetimes, seiz_types)
        window["-SEIZURE SELECTION-"].update(values=seizures_selection_list)

        # Plot the latent space with default values
        curr_canvas_agg_tab1, curr_canvas_agg_tab2 = plot_latent_on_canvas(window, 
                    curr_canvas_agg_tab1=curr_canvas_agg_tab1,
                    curr_canvas_agg_tab2=curr_canvas_agg_tab2,
                    latent_data=latent_data,
                    win_style=DEFAULT_WIN_STYLE,
                    samp_freq=SAMP_FREQ,
                    start_datetime=abs_start_datetime, 
                    stop_datetime=abs_stop_datetime, 
                    abs_start_datetime=abs_start_datetime,
                    abs_stop_datetime=abs_stop_datetime,
                    win_sec=DEFAULT_WIN,
                    stride_sec=DEFAULT_STRIDE,
                    seiz_start_dt=pat_seiz_start_datetimes,
                    seiz_stop_dt=pat_seiz_stop_datetimes,
                    preictal_dur=DEFAULT_PREICTAL_DUR,
                    postictal_dur=DEFAULT_POSTICTAL_DUR,
                    absolute_latent=ABSOLUTE_LATENT,
                    max_latent=MAX_LATENT,
                    auto_scale_plot=DEFAULT_AUTO_SCALE_PLOT,
                    interictal_contour=DEFAULT_INTERICTAL_CONTOUR,
                    tab2_pickle_file=tab2_pickle_file)
                        
        
        # Update the input fields with default start and end times

    elif event == '-UPDATE BUTTON-':
        
        s =  values['-START TIMESTAMP INPUT-']
        new_start_datetime = datetime.datetime(int(s[6:10]),int(s[0:2]),int(s[3:5]),int(s[11:13]),int(s[14:16]),int(s[17:19]))
        s =  values['-STOP TIMESTAMP INPUT-']
        new_stop_datetime = datetime.datetime(int(s[6:10]),int(s[0:2]),int(s[3:5]),int(s[11:13]),int(s[14:16]),int(s[17:19]))
        new_win_sec = float(values['-WINDOW DUR-'])
        new_stride = float(values['-STRIDE-'])
        new_preictal = float(values['-PREICTAL DUR-'])
        new_postictal = float(values['-POSTICTAL DUR-'])
        abs_val = values['-ABS LATENT-']
        max_val = values['-MAX LATENT-']
        auto_scale_plot = values['-AUTO SCALE-']
        # win_style = values['-WIN STYLE-']
        fbtc_bool=values['-FBTC-']
        fias_bool=values['-FIAS-']
        fas2fias_bool=values['-FAStoFIAS-']
        fas_bool=values['-FAS-']
        focal_unknown_bool=values['-FOCAL UNKNOWN-']
        subclinical_bool=values['-SUBCLINICAL-']
        unknown_bool=values['-UNKNOWN TYPE-']
        non_electro_bool=values['-NONELECTRO TYPE-']
        interictal_contour=values['-INTERICTAL CONTOUR-']

        pat_seiz_start_datetimes, pat_seiz_stop_datetimes, seiz_types = utils_functions.get_pat_seiz_datetimes(
            pat_id=pat_id,
            FBTC_bool=fbtc_bool,
            FIAS_bool=fias_bool,
            FAS_to_FIAS_bool=fas2fias_bool,
            FAS_bool=fas_bool,
            focal_unknown_bool=focal_unknown_bool,
            subclinical_bool=subclinical_bool,
            unknown_bool=unknown_bool,
            non_electro_bool=non_electro_bool)
        
        seizures_selection_list = compile_seiz_list(pat_seiz_start_datetimes,pat_seiz_stop_datetimes, seiz_types)
        window["-SEIZURE SELECTION-"].update(values=seizures_selection_list)

        curr_canvas_agg_tab1, curr_canvas_agg_tab2 = plot_latent_on_canvas(window, 
                    curr_canvas_agg_tab1=curr_canvas_agg_tab1,
                    curr_canvas_agg_tab2=curr_canvas_agg_tab2,
                    latent_data=latent_data,
                    win_style=DEFAULT_WIN_STYLE,
                    samp_freq=SAMP_FREQ,
                    start_datetime=[new_start_datetime], 
                    stop_datetime=[new_stop_datetime], 
                    abs_start_datetime=abs_start_datetime,
                    abs_stop_datetime=abs_stop_datetime,
                    win_sec=new_win_sec,
                    stride_sec=new_stride,
                    seiz_start_dt=pat_seiz_start_datetimes,
                    seiz_stop_dt=pat_seiz_stop_datetimes,
                    preictal_dur=new_preictal,
                    postictal_dur=new_postictal,
                    absolute_latent=abs_val,
                    max_latent=max_val,
                    auto_scale_plot=auto_scale_plot,
                    interictal_contour=interictal_contour,
                    tab2_pickle_file=tab2_pickle_file)
   
    elif event == '-RESET BUTTON-':
        
        window['-START TIMESTAMP INPUT-'].update(abs_start_datetime[0].strftime("%m/%d/%Y-%H:%M:%S"))
        window['-STOP TIMESTAMP INPUT-'].update(abs_stop_datetime[0].strftime("%m/%d/%Y-%H:%M:%S"))
        window['-WINDOW DUR-'].update(str(DEFAULT_WIN))
        window['-STRIDE-'].update(str(DEFAULT_STRIDE))
        window['-ABS LATENT-'].update(value=ABSOLUTE_LATENT)
        window['-MAX LATENT-'].update(value=MAX_LATENT)
        window['-AUTO SCALE-'].update(value=DEFAULT_AUTO_SCALE_PLOT)
        # window["-WIN STYLE-"].update(value=DEFAULT_WIN_STYLE)
        window['-PREICTAL DUR-'].update(str(DEFAULT_PREICTAL_DUR))
        window['-POSTICTAL DUR-'].update(str(DEFAULT_POSTICTAL_DUR))
        window["-FBTC-"].update(value=DEFAULT_FBTC)
        window["-FIAS-"].update(value=DEFAULT_FIAS)
        window["-FAStoFIAS-"].update(value=DEFAULT_FAStoFIAS)
        window["-FAS-"].update(value=DEFAULT_FAS)
        window['-FOCAL UNKNOWN-'].update(value=DEFAULT_FOCAL_UNKNOWN)
        window["-SUBCLINICAL-"].update(value=DEFAULT_SUBCLINICAL)
        window["-UNKNOWN TYPE-"].update(value=DEFAULT_UNKNOWN)
        window["-NONELECTRO TYPE-"].update(value=DEFAULT_NONELECTRO)
        window["-INTERICTAL CONTOUR-"].update(value=DEFAULT_INTERICTAL_CONTOUR)
        
        pat_seiz_start_datetimes, pat_seiz_stop_datetimes, seiz_types = utils_functions.get_pat_seiz_datetimes(
            pat_id=pat_id,
            FBTC_bool=DEFAULT_FBTC,
            FIAS_bool=DEFAULT_FIAS,
            FAS_to_FIAS_bool=DEFAULT_FAStoFIAS,
            FAS_bool=DEFAULT_FAS,
            focal_unknown_bool=DEFAULT_FOCAL_UNKNOWN,
            subclinical_bool=DEFAULT_SUBCLINICAL,
            unknown_bool=DEFAULT_UNKNOWN,
            non_electro_bool=DEFAULT_NONELECTRO
            )

        seizures_selection_list = compile_seiz_list(pat_seiz_start_datetimes,pat_seiz_stop_datetimes, seiz_types)
        window["-SEIZURE SELECTION-"].update(values=seizures_selection_list)

        # Plot the latent space with default values
        curr_canvas_agg_tab1, curr_canvas_agg_tab2 = plot_latent_on_canvas(window, 
                    curr_canvas_agg_tab1=curr_canvas_agg_tab1,
                    curr_canvas_agg_tab2=curr_canvas_agg_tab2,
                    latent_data=latent_data,
                    win_style=DEFAULT_WIN_STYLE,
                    samp_freq=SAMP_FREQ,
                    start_datetime=abs_start_datetime, 
                    stop_datetime=abs_stop_datetime, 
                    abs_start_datetime=abs_start_datetime,
                    abs_stop_datetime=abs_stop_datetime,
                    win_sec=DEFAULT_WIN,
                    stride_sec=DEFAULT_STRIDE,
                    seiz_start_dt=pat_seiz_start_datetimes,
                    seiz_stop_dt=pat_seiz_stop_datetimes,
                    preictal_dur=DEFAULT_PREICTAL_DUR,
                    postictal_dur=DEFAULT_POSTICTAL_DUR,
                    absolute_latent=ABSOLUTE_LATENT,
                    max_latent=MAX_LATENT,
                    auto_scale_plot=DEFAULT_AUTO_SCALE_PLOT,
                    interictal_contour=DEFAULT_INTERICTAL_CONTOUR,
                    tab2_pickle_file=tab2_pickle_file)
                    
window.close()