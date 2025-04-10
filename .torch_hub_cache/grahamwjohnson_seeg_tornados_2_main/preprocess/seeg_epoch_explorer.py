msg = 'Hello, World!'
print(msg)

import PySimpleGUI as sg
import os.path
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl
pl.switch_backend('agg')
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

SAMP_FREQ = 512 # assumed

SAMPLES_TO_PLOT = 512*2

# BASE_SEARCH_PATH = '/media/graham/MOBO_RAID0/Ubuntu_Projects/SEEG_Tornados/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/DurStr_1024s896s_epoched_datasets'
BASE_SEARCH_PATH = '/media/glommy1/data/vanderbilt_seeg/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_normalized_to_first_24_hours/wholeband/DurStr_1024s896s_epoched_datasets'

# First the window layout in 2 columns
ch_names_prev = -1

def filename_to_datetimes(list_file_names):
        start_datetimes = [datetime.datetime.min]*len(list_file_names)
        stop_datetimes = [datetime.datetime.min]*len(list_file_names)
        for i in range(0, len(list_file_names)):
            splits = list_file_names[i].split('_')
            aD = splits[1]
            aT = splits[2]
            start_datetimes[i] = datetime.datetime(int(aD[4:8]), int(aD[0:2]), int(aD[2:4]), int(aT[0:2]), int(aT[2:4]), int(aT[4:6]), int(int(aT[6:8])*1e4))
            bD = splits[4]
            bT = splits[5]
            stop_datetimes[i] = datetime.datetime(int(bD[4:8]), int(bD[0:2]), int(bD[2:4]), int(bT[0:2]), int(bT[2:4]), int(bT[4:6]), int(int(bT[6:8])*1e4))
        return start_datetimes, stop_datetimes

def draw_figure(canvas, figure):
   figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
   figure_canvas_agg.draw()
   figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
   return figure_canvas_agg

file_list_column = [
    [
        sg.Text("Pickle Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(initial_folder=BASE_SEARCH_PATH),
    ],
    [
        #sg.Text("Epoch File List"),
        sg.Listbox(
            values=[], enable_events=True, size=(70, 20), key="-FILE LIST-"
        )
    ],
    [
        #sg.Text("Channel List"),
        sg.Listbox(
            values=[], enable_events=True, size=(70, 20), key="-CHANNEL LIST-"
        )
    ],
    [
        sg.Input('', enable_events=True, size=(30, 5), key="-START TIMESTAMP INPUT-"),
        sg.Text("Start Timestamp"),
        sg.Button('Update Time', key="-UPDATE TIME BUTTON-")
    ],
    [
        sg.Input('', enable_events=True, size=(30, 5), key="-START SAMPLE IDX-"),
        sg.Text("Start Sample Index"),
        sg.Button('Update Sample', key="-UPDATE SAMPLE BUTTON-")
    ],
    [
        sg.Input(key='-FILENAME-', visible=False, enable_events=True), 
        sg.FileSaveAs(),
        sg.Button('Save', key="-SAVE BUTTON-")
    ]
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Browse to epoch directory\n\nChoose a file\n\nThen select channel to display", key="-INSTRUCTIONS-")],
    [sg.Text(size=(100, 1), key="-TOUT-")],
    [sg.Canvas(key="-CANVAS-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

# Initialize the GUI
window = sg.Window("SEEG Epoch Explorer (SEE)", layout)
selected_ch = 'CH_0'

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:

        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".pkl"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        # try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )

            # Load in the pickle file
            file = open(filename,'rb')
            data = pickle.load(file)
            file.close()

            # ch_names = str(1:data.shape[0])
            ch_names = ["CH_" + str(item) for item in range(0, data.shape[0])]
            if ch_names != ch_names_prev:
                selected_ch = 'CH_0'
                selected_ch_index = 0
                window["-CHANNEL LIST-"].update(ch_names)
                window["-CHANNEL LIST-"].update(set_to_index=[selected_ch_index], scroll_to_index=selected_ch_index)
                ch_names_prev = ch_names

            # Plotting code 
            start_sample = 0
            selected_ch_index = int(selected_ch.split("_")[1])
            fig = matplotlib.figure.Figure(figsize=(6, 4), dpi=100)
            plot_sample_range = np.linspace(0,SAMPLES_TO_PLOT,SAMPLES_TO_PLOT, dtype=int)
            x_seconds = plot_sample_range/SAMP_FREQ
            y_scaled = data[selected_ch_index,start_sample:start_sample + SAMPLES_TO_PLOT]
            fig.add_subplot(111).plot(x_seconds,y_scaled,label="Scaled")
            fig.axes[0].set_ylim([-1, 1])
            fig.axes[0].set_ylabel("Amplitude, normalized to first X hours in EMU")
            fig.axes[0].set_xlabel("Time, seconds")
            window["-TOUT-"].update(filename)
            window["-INSTRUCTIONS-"]('')
            if 'fig_canvas_agg' in locals(): fig_canvas_agg.get_tk_widget().forget()
            fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
            currr_start_dt, _ = filename_to_datetimes(values["-FILE LIST-"])
            window['-START TIMESTAMP INPUT-'].update(currr_start_dt[0].strftime("%m/%d/%Y-%H:%M:%S.%f"))
            window['-START SAMPLE IDX-'].update(f'{start_sample}')

        # except:
        #     pass

    
    elif event == "-CHANNEL LIST-":  # A new channel was chosen from the listbox 
        #try:
            # Plotting code 
            selected_ch = values["-CHANNEL LIST-"][0]
            selected_ch_index = int(selected_ch.split("_")[1])
            start_sample =  int(values['-START SAMPLE IDX-'])
            fig = matplotlib.figure.Figure(figsize=(6, 4), dpi=100)
            plot_sample_range = np.linspace(0,SAMPLES_TO_PLOT, SAMPLES_TO_PLOT, dtype=int)
            x_seconds = plot_sample_range/SAMP_FREQ
            y_scaled = data[selected_ch_index,start_sample:start_sample + SAMPLES_TO_PLOT]
            fig.add_subplot(111).plot(x_seconds,y_scaled,label="Scaled")
            fig.axes[0].set_ylim([-1, 1])
            fig.axes[0].set_ylabel("Amplitude, normalized to first X hours in EMU")
            fig.axes[0].set_xlabel("Time, seconds")
            window["-TOUT-"].update(filename)
            window["-INSTRUCTIONS-"]('')
            if 'fig_canvas_agg' in locals(): fig_canvas_agg.get_tk_widget().forget()
            fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
            currr_start_dt, _ = filename_to_datetimes(values["-FILE LIST-"])
            window['-START TIMESTAMP INPUT-'].update(currr_start_dt[0].strftime("%m/%d/%Y-%H:%M:%S.%f"))
            window['-START SAMPLE IDX-'].update(f'{start_sample}')

        #except:
            #pass
    elif event == '-UPDATE TIME BUTTON-':
            s =  values['-START TIMESTAMP INPUT-']
            new_start_datetime = datetime.datetime(int(s[6:10]),int(s[0:2]),int(s[3:5]),int(s[11:13]),int(s[14:16]),int(s[17:19]))
            selected_ch = values["-CHANNEL LIST-"][0]
            selected_ch_index = int(selected_ch.split("_")[1])
            fig = matplotlib.figure.Figure(figsize=(6, 4), dpi=100)
            plot_sample_range = np.linspace(0,SAMPLES_TO_PLOT, SAMPLES_TO_PLOT, dtype=int)
            x_seconds = plot_sample_range/SAMP_FREQ

            # Shift the plot to the desired start time
            file_start_datetime, _ = filename_to_datetimes(values["-FILE LIST-"])
            start_sample = int((new_start_datetime - file_start_datetime[0]).total_seconds()*SAMP_FREQ)
            y_scaled = data[selected_ch_index, start_sample:start_sample + SAMPLES_TO_PLOT] # Index into proper spot in file

            fig.add_subplot(111).plot(x_seconds,y_scaled,label="Scaled")
            fig.axes[0].set_ylim([-1, 1])
            fig.axes[0].set_ylabel("Amplitude, normalized to first X hours in EMU")
            fig.axes[0].set_xlabel("Time, seconds")
            window["-TOUT-"].update(filename)
            window["-INSTRUCTIONS-"]('')
            if 'fig_canvas_agg' in locals(): fig_canvas_agg.get_tk_widget().forget()
            fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
            window['-START SAMPLE IDX-'].update(f'{start_sample}')

    elif event == '-UPDATE SAMPLE BUTTON-':
            start_sample =  int(values['-START SAMPLE IDX-'])
            selected_ch = values["-CHANNEL LIST-"][0]
            selected_ch_index = int(selected_ch.split("_")[1])
            fig = matplotlib.figure.Figure(figsize=(6, 4), dpi=100)
            plot_sample_range = np.linspace(0,SAMPLES_TO_PLOT, SAMPLES_TO_PLOT, dtype=int)
            x_seconds = plot_sample_range/SAMP_FREQ

            # Shift the plot to the desired SAMPLE
            y_scaled = data[selected_ch_index, start_sample:start_sample + SAMPLES_TO_PLOT] # Index into proper spot in file

            fig.add_subplot(111).plot(x_seconds,y_scaled,label="Scaled")
            fig.axes[0].set_ylim([-1, 1])
            fig.axes[0].set_ylabel("Amplitude, normalized to first X hours in EMU")
            fig.axes[0].set_xlabel("Time, seconds")
            window["-TOUT-"].update(filename)
            window["-INSTRUCTIONS-"]('')
            if 'fig_canvas_agg' in locals(): fig_canvas_agg.get_tk_widget().forget()
            fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
            file_start_dt, _ = filename_to_datetimes(values["-FILE LIST-"])
            curr_start_dt = file_start_dt[0] + datetime.timedelta(seconds=start_sample/SAMP_FREQ)
            window['-START TIMESTAMP INPUT-'].update(curr_start_dt.strftime("%m/%d/%Y-%H:%M:%S.%f"))

    elif event == '-SAVE BUTTON-':
            
            filename = values['-FILENAME-']
            with open(filename, 'w') as f:
                
                start_sample =  int(values['-START SAMPLE IDX-'])
                selected_ch = values["-CHANNEL LIST-"][0]
                selected_ch_index = int(selected_ch.split("_")[1])
                
                plot_sample_range = np.linspace(0,SAMPLES_TO_PLOT, SAMPLES_TO_PLOT, dtype=int)
                x_seconds = plot_sample_range/SAMP_FREQ

                # Shift the plot to the desired SAMPLE
                y_scaled = data[selected_ch_index, start_sample:start_sample + SAMPLES_TO_PLOT] # Index into proper spot in file

                gs = gridspec.GridSpec(1, 5)
                fig = plt.figure(figsize=(20, 14))
                fig.add_subplot(111).plot(x_seconds,y_scaled,label="Scaled")

                plt.savefig(f"{filename}.jpg", dpi=400)
                plt.savefig(f"{filename}.svg")
                plt.close(fig) 
            

window.close()