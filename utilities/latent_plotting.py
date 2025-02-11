import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import datetime
import pandas as pd
import multiprocessing as mp
import functools
import seaborn as sns
import matplotlib as mpl

# Turn of interactive polotting for speed
plt.ioff()

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def plot_latent(
    ax, 
    interCont_ax, 
    seiztype_ax, 
    time_ax, 
    cluster_ax,  
    latent_data, 
    modified_samp_freq, 
    start_datetimes, 
    stop_datetimes, 
    win_sec, 
    stride_sec, 
    seiz_start_dt, 
    seiz_stop_dt, 
    seiz_types,
    preictal_dur, 
    postictal_dur, 
    seiz_type_list, 
    seiz_plot_mult, 
    hdb_labels, 
    hdb_probabilities, 
    hdb,
    tab2_lighten=False, 
    plot_alpha=0.5, 
    plot_alpha_TIME=0.3, 
    s_plot=15, 
    SPES_colorbar=False, 
    auto_scale_plot=True, 
    xy_lims = [],
    plot_ictal=True, 
    **kwargs): 
    
    if len(latent_data.shape) != 3: raise Exception("Must pass 3D latent data [epoch, 2, timesample]")

    # Sort the files based on datetime
    sort_idxs = sorted(range(len(start_datetimes)), key=lambda k: start_datetimes[k])
    start_datetimes_sorted = [start_datetimes[sort_idx] for sort_idx in sort_idxs]
    stop_datetimes_sorted = [stop_datetimes[sort_idx] for sort_idx in sort_idxs]
    latent_data_sorted = latent_data[sort_idxs, :, :]  # can use direct indexing because it's a numpy array
    hdb_labels_sorted = hdb_labels[sort_idxs, :, :]
    hdb_probabilities = hdb_probabilities[sort_idxs, :, :]

    lat_data_windowed_toplot = np.empty([2,0])
    c_toplot = np.empty([2,0])
    c_ST_toplot = np.empty([2,0])
    c_CLUSTER_toplot = np.empty([2,0])

    for iii in range(0,latent_data.shape[0]):

        # # Pull the proper data (with desired step size - averaged over step) out of main latent variable
        # samp_idxs, x_datetimes, lat_data_windowed = get_latent_datapoints(
        #     data=latent_data_sorted[iii, :, :], 
        #     win_style=win_style,
        #     modified_samp_freq=modified_samp_freq, 
        #     start_datetimes_sorted=start_datetimes_sorted[iii], 
        #     stop_datetimes_sorted=stop_datetimes_sorted[iii], 
        #     abs_start_datetime=start_datetimes_sorted[iii], 
        #     abs_stop_datetime=stop_datetimes_sorted[iii],
        #     win_sec=win_sec,
        #     stride_sec=stride_sec, 
        #     absolute_latent=absolute_latent,
        #     max_latent=max_latent)

        # Get the datetimes for every single datapoint
        # IMPORTANT: the datetime is at the END of the window, so a given datetime represents the PAST data 
        lat_data_windowed = latent_data_sorted[iii, :, :]
        start_dt_curr = start_datetimes_sorted[iii]
        x_datetimes = [start_dt_curr + datetime.timedelta(seconds=win_sec + samp / modified_samp_freq) for samp in range(lat_data_windowed.shape[1])]
        samp_idxs = np.arange(0, lat_data_windowed.shape[1])

        # Color by inter-ictal, pre-ictal, ictal, post-ictal
        # ASSUMES A CYCLICAL COLOR MAP HAS BEEN CHOSEN
        c_interictal_val_MIN = -1
        c_preictal_max_val = -0.25
        c_ictal_val = 0
        c_postictal_min_val =  0.25
        c_interictal_val_MAX = 1
        preictal_sec = preictal_dur
        postictal_sec = postictal_dur

        # Initialize colorbars to all -1
        c = np.ones(len(x_datetimes)) * c_interictal_val_MIN
        c_ST = np.ones(len(x_datetimes)) * c_interictal_val_MIN

        # Calculate colors for each subplot type
        for i in range(0, len(seiz_start_dt[iii])):
            # THIS WILL BE POSITIVE IF ANY PART OF AVERAGING WINDOW IS ICTAL
            x_win_ictal_bool_curr = [(d >= seiz_start_dt[iii][i]) & (d - datetime.timedelta(seconds=win_sec) <= seiz_stop_dt[iii][i]) for d in x_datetimes] 
            
            # WILL BE POSITIVE IF leading edge of sliding window hits preictal period, but has not entered ictal period AT ALL
            # Add a 2 second buffer to account for coloring gap (any ictal encroaching will be overwtitten by ictal later)
            x_win_preictal_bool_curr = [(d > (seiz_start_dt[iii][i] - datetime.timedelta(seconds=preictal_sec))) & (d < seiz_start_dt[iii][i] + datetime.timedelta(seconds=2)) for d in x_datetimes]
            x_win_preictal_IDXs = [i for i, x in enumerate(x_win_preictal_bool_curr) if x]
            
            # WIll be Positive if TRAILING edge of window is out of ictal period and TRAILING edge is within postictal seconds desired
            x_win_postictal_bool_curr = [(d - datetime.timedelta(seconds=win_sec) > seiz_stop_dt[iii][i]) & (d - datetime.timedelta(seconds=win_sec) < seiz_stop_dt[iii][i] + datetime.timedelta(seconds=postictal_sec)) for d in x_datetimes]
            x_win_postictal_IDXs = [i for i, x in enumerate(x_win_postictal_bool_curr) if x] 

            # Update colors if this seizure is in plot's time range.
            # Blend pre and post ictal colors from previous seizures
            # Prioritize color override as ictal > preictyal > postictal

            # Seiztype colorvals, cyclical surrounding CHANGES based on seiz_type
            curr_seiz_type = seiz_types[iii][i]
            seiz_type_shiftval = seiz_plot_mult[seiz_type_list.index(curr_seiz_type)]
            c_ST_interictal_val_MIN = 0 + seiz_type_shiftval
            c_ST_preictal_max_val = 0.25 + seiz_type_shiftval
            c_ST_ictal_val = 1 + seiz_type_shiftval
            c_ST_postictal_min_val =  1.25 + seiz_type_shiftval
            c_ST_interictal_val_MAX = 2 + seiz_type_shiftval

            # Are there any postictal timepoints 
            if x_win_postictal_bool_curr.count(True) > 0:
                # FIRST update the POSTICTAL values if the value for the CURRENT seizure is GREATER than current value (cyclic color scale with 0=ictal)
                c_postictal_taper_vals = np.linspace(c_postictal_min_val, c_interictal_val_MAX, len(x_win_postictal_IDXs))
                c_ST_postictal_taper_vals = np.linspace(c_ST_postictal_min_val, c_ST_interictal_val_MAX, len(x_win_postictal_IDXs))
                count_c = 0
                for c_idx in range(x_win_postictal_IDXs[0], x_win_postictal_IDXs[-1]):
                    if c_postictal_taper_vals[count_c] > c[c_idx]:
                        c[c_idx] = c_postictal_taper_vals[count_c]
                        c_ST[c_idx] = c_ST_postictal_taper_vals[count_c]
                    count_c += 1

            # Are there any preictal timepoints?
            if x_win_preictal_bool_curr.count(True) > 0:
                # SECOND update and override for the PREICTAL values if the value for the CURRENT seizure is GREATER than current value (cyclic color scale with 0=ictal)
                c_preictal_taper_vals = np.linspace(c_interictal_val_MIN, c_preictal_max_val, len(x_win_preictal_IDXs))
                c_ST_preictal_taper_vals = np.linspace(c_ST_interictal_val_MIN, c_ST_preictal_max_val, len(x_win_preictal_IDXs))
                count_c = 0
                for c_idx in range(x_win_preictal_IDXs[0], x_win_preictal_IDXs[-1]):
                    if c[c_idx] > 0: # Alreday been a postictal coloring done here
                        c[c_idx] = c_preictal_taper_vals[count_c]
                        c_ST[c_idx] = c_ST_preictal_taper_vals[count_c]
                    elif c_preictal_taper_vals[count_c] > c[c_idx]: # Now check if no other evernt has colored this more... don't know how it would have
                        c[c_idx] = c_preictal_taper_vals[count_c]
                        c_ST[c_idx] = c_ST_preictal_taper_vals[count_c]
                    count_c += 1

            # Are there any ictal timepoints?  
            if x_win_ictal_bool_curr.count(True) > 0:
                # THIRD: Ictal will override everything, but pre and post will be "blended" with adjacent seizure tapers if pre/post periods are set long enough
                c[x_win_ictal_bool_curr] = c_ictal_val
                c_ST[x_win_ictal_bool_curr] = c_ST_ictal_val

        # DELETE SEIZURES (if selected)
        title_ictal_included = " with Seizures Plotted"
        if not plot_ictal:
            x_datetimes = np.array(x_datetimes)[c != 0].tolist()
            lat_data_windowed = lat_data_windowed[:, c != 0]
            c = c[c != 0]
            c_ST = c_ST[c != 0]
            title_ictal_included = " without Seizures Plotted"

        # Collect each epochs data and color scheme
        if lat_data_windowed.shape[1] != c.shape[0]: raise Exception("Arrays have different number of time samples")
        lat_data_windowed_toplot = np.column_stack([lat_data_windowed_toplot, lat_data_windowed])
        c_toplot = np.append(c_toplot, c)
        c_ST_toplot = np.append(c_ST_toplot, c_ST)
        c_CLUSTER_toplot = np.append(c_CLUSTER_toplot, hdb_labels_sorted[iii, 0, samp_idxs-1])

        
    if not SPES_colorbar:

        # *** PERI-ICTAL LABELED PLOT ***

        if tab2_lighten:
            cmap = plt.cm.twilight_r
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmaplist[0] = (1.0,1.0,1.0,1.0)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)
            cmap = cmap_map(lambda x: x/5 + 0.7, cmap)
        else:
            cmap = plt.get_cmap('twilight_r')

        # Before plotting, order by color to have pre/ictal/post on top of interictal
        plot_order = np.argsort(-1 * np.abs(c_toplot)) 

        x_plot = lat_data_windowed_toplot[0,:]
        y_plot = lat_data_windowed_toplot[1,:]
        s = ax.scatter(x_plot[plot_order], y_plot[plot_order], c=c_toplot[plot_order], alpha=plot_alpha, s=s_plot, cmap=cmap, edgecolors='none', vmin=c_interictal_val_MIN, vmax=c_interictal_val_MAX)
        
        # Peri-ictal colorbar
        cbar = plt.colorbar(
            ax.collections[0], 
            ax=ax, 
            ticks=[c_interictal_val_MIN,
            c_interictal_val_MIN/2, 
            0,
            c_interictal_val_MAX/2,
            c_interictal_val_MAX], 
            orientation='horizontal',
            shrink=0.7)
        cbar.ax.set_xticklabels(['Interictal', 'Preictal', 'Ictal', 'Postictal', 'Interictal'])
        # cbar.ax.set_title('Peri-Ictal Labels')

        # Scale the plot
        ax.set_title('Latent Space' + title_ictal_included)
        ax.set_ylabel("Latent Var 1")
        ax.set_xlabel("Latent Var 0")
        ax.set_aspect('equal')
        if not auto_scale_plot:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        elif xy_lims != []:
            ax.set_xlim(xy_lims[0])
            ax.set_ylim(xy_lims[1])
        else:
            max_x_lim = np.max(np.abs(ax.get_xlim()))
            max_y_lim = np.max(np.abs(ax.get_ylim()))
            marg_x = (np.nanmax(x_plot) - np.nanmin(x_plot)) * ax.margins()[0]
            marg_y = (np.nanmax(y_plot) - np.nanmin(y_plot)) * ax.margins()[1]
            max_x_new = np.nanmax(np.abs(x_plot)) + marg_x
            max_y_new = np.nanmax(np.abs(y_plot)) + marg_y
            max_x = -1
            max_y = -1
            if max_x_new > max_x_lim: max_x = max_x_new
            else: max_x = max_x_lim
            if max_y_new > max_y_lim: max_y = max_y_new
            else: max_y = max_y_lim
            max_xy = np.max([max_x, max_y])
            ax.set_xlim(-max_xy, max_xy)
            ax.set_ylim(-max_xy, max_xy)


        # *** COUNTOUR PLOT ***
            
        x_plot_contour = x_plot[np.abs(c_toplot) ==1]
        y_plot_contour = y_plot[np.abs(c_toplot) ==1]
        cbar_dict = {'location': 'bottom', 'orientation': 'horizontal', 'label': 'Interictal Density', 'format': '%.2e'}
        s = sns.kdeplot(x=x_plot_contour, y=y_plot_contour, ax=interCont_ax, cmap="Greys", fill=True, bw_adjust=.5, cbar=True, cbar_kws=cbar_dict)
        # plt.colorbar(shrink=0.7) 
        # Reset the limits based on peri-ictal plot
        interCont_ax.set_xlim(ax.get_xlim())
        interCont_ax.set_ylim(ax.get_ylim())
        interCont_ax.set_ylabel("Latent Var 1")
        interCont_ax.set_xlabel("Latent Var 0")
        interCont_ax.set_aspect('equal')


        # *** SEIZ TYPE PLOT ***

        # Special seiztype color scheme
        bounds = [-1] + seiz_plot_mult + [max(seiz_plot_mult) + 2]

        #              ['FBTC',    'FIAS',    'FAS_to_FIAS', 'FAS',     'Focal unknown awareness', 'Unknown', 'Subclinical', 'Non-electrographic'
        cmaps_to_use = ['Purples', 'Reds',    'Oranges',     'Blues',   'BrBG_r',                   'BrBG_r',   'Greens',      'pink_r']
        samp_vec = np.arange(0.75, 0.76, 0.01)
        # initialize interictals with twilight(0)
        cmap_twilight = plt.get_cmap('twilight')
        cmap_ST_list = [cmap_twilight(0) for i in range(2 * len(samp_vec))] 
        # Iterate through colorbars
        for i in range(len(cmaps_to_use)):
            # Pull out cbar values
            cmap_curr = plt.get_cmap(cmaps_to_use[i])
            vals = cmap_curr(samp_vec).tolist()
            # Flip the colorbar and reverse to get cyclical map
            vals_r = vals.copy()
            vals_r.reverse()
            vals_bi = vals + vals_r
            
            # Add to sequential master colorbar
            cmap_ST_list = cmap_ST_list + vals_bi

        cmap_ST = mpl.colors.ListedColormap(cmap_ST_list)

        # Before plotting, order by color to have pre/ictal/post on top of interictal
        plot_order_ST = np.argsort(c_ST_toplot)
        x_plot_ST = lat_data_windowed_toplot[0,:]
        y_plot_ST = lat_data_windowed_toplot[1,:]        
        s_ST = seiztype_ax.scatter(
            x_plot_ST[plot_order_ST], y_plot_ST[plot_order_ST], c=c_ST_toplot[plot_order_ST], 
            alpha=plot_alpha, s=s_plot, cmap=cmap_ST, edgecolors='none', vmin=bounds[0], vmax=bounds[-1]) 
        # Reset the limits based on peri-ictal plot
        seiztype_ax.set_xlim(ax.get_xlim())
        seiztype_ax.set_ylim(ax.get_ylim())
        seiztype_ax.set_ylabel("Latent Var 1")
        seiztype_ax.set_xlabel("Latent Var 0")
        seiztype_ax.set_aspect('equal')

        # norm = mpl.colors.Normalize(vmin=5, vmax=10)
        ticks = [e + 1 for e in bounds[:-1]]
        labels = ['Interictal'] + seiz_type_list
        cbar_ST = plt.colorbar(seiztype_ax.collections[0], ax=seiztype_ax, cmap=cmap_ST, orientation='horizontal', ticks=ticks, shrink=0.7)
        cbar_ST.ax.set_xticklabels(labels)


        # *** TIME PLOTTING ***

        cmap_time_raw = plt.get_cmap('cubehelix')
        samp_vec_time = np.arange(0.0, 0.9, 0.01)
        cmap_time_list = cmap_time_raw(samp_vec_time) 
        cmap_time = mpl.colors.ListedColormap(cmap_time_list)

        x_plot_time = lat_data_windowed_toplot[0,:]
        y_plot_time = lat_data_windowed_toplot[1,:]  
        s_time = time_ax.scatter(
            x_plot_time, y_plot_time, c=np.linspace(0, 1, len(x_plot_time)).tolist(), 
            alpha=plot_alpha_TIME, s=s_plot, cmap=cmap_time, edgecolors='none', vmin=0, vmax=1) 
        
        # Colorbar: Find the number of midnights and set as x-axis ticks at each midnight's percentage of total EMU time
        midnights_perc_list = []
        midnight_count = 0
        midnight_label = []
        total_seconds_EMU = (stop_datetimes_sorted[-1] - start_datetimes_sorted[0]).total_seconds()
        for i in range(len(start_datetimes_sorted)):
            if start_datetimes_sorted[i].day != stop_datetimes_sorted[i].day:
                midnight_datetime = stop_datetimes_sorted[i].replace(hour=0, minute=0, second=0, microsecond=0)
                midnight_count = midnight_count + 1
                midnight_perc_curr = (midnight_datetime - start_datetimes_sorted[0]).total_seconds() / total_seconds_EMU
                if midnights_perc_list == []:
                    midnights_perc_list = [midnight_perc_curr]
                    midnight_label = [str(midnight_count)]
                else: 
                    midnights_perc_list = midnights_perc_list + [midnight_perc_curr]
                    midnight_label = midnight_label + [str(midnight_count)]

        cbar_time = plt.colorbar(time_ax.collections[0], ax=time_ax, cmap=cmap_time, orientation='horizontal', ticks=midnights_perc_list, shrink=0.7, label='Time (Midnight Count)')
        cbar_time.ax.set_xticklabels(midnight_label)
        # Reset the limits based on peri-ictal plot
        time_ax.set_xlim(ax.get_xlim())
        time_ax.set_ylim(ax.get_ylim())
        time_ax.set_ylabel("Latent Var 1")
        time_ax.set_xlabel("Latent Var 0")
        time_ax.set_aspect('equal')



        # **** HDBSCAN CLUSTER PLOTTING ****

        plot_order_CLUSTER = np.argsort(c_CLUSTER_toplot)
        cmap_CLUSTER = plt.get_cmap('gist_ncar_r')
        # pal = sns.color_palette('deep', 8)
        # colors = [sns.desaturate(pal[col], sat) for col, sat in zip(hdb_labels_allFiles_expanded[plot_order_CLUSTER],
        #                                                     hdb_probabilities_allFiles_expanded[plot_order_CLUSTER])]
        x_plot_CLUSTER = lat_data_windowed_toplot[0,:]
        y_plot_CLUSTER = lat_data_windowed_toplot[1,:]        
        s_CLUSTER = cluster_ax.scatter(
            x_plot_CLUSTER[plot_order_CLUSTER], y_plot_CLUSTER[plot_order_CLUSTER], c=c_CLUSTER_toplot[plot_order_CLUSTER], 
            alpha=0.25, s=s_plot, cmap=cmap_CLUSTER, edgecolors='none', vmin=-3, vmax=np.max(hdb.labels_)) #, vmin=bounds[0], vmax=bounds[-1]) 
        # Reset the limits based on peri-ictal plot
        cluster_ax.set_xlim(ax.get_xlim())
        cluster_ax.set_ylim(ax.get_ylim())
        cluster_ax.set_ylabel("Latent Var 1")
        cluster_ax.set_xlabel("Latent Var 0")
        cluster_ax.set_aspect('equal')

        # norm = mpl.colors.Normalize(vmin=5, vmax=10)
        # ticks = [e + 1 for e in bounds[:-1]]
        # labels = ['Interictal'] + seiz_type_list
        cbar_CLUSTER = plt.colorbar(cluster_ax.collections[0], ax=cluster_ax, cmap=cmap_CLUSTER, orientation='horizontal', shrink=0.7, label='Cluster Index')
        # cbar_CLUSTER.ax.set_xticklabels(labels)



        # DEBUGGING
        # import os
        # plt.savefig(f"{os.getcwd()}/test.jpg")

        # plt.close()
        # fig = plt.figure(figsize=(26, 26))
        # gs = gridspec.GridSpec(2, 2, figure=fig)
        # ax = fig.add_subplot(gs[0, 0]) 
        # interCont_ax = fig.add_subplot(gs[0, 1]) 
        # seiztype_ax = fig.add_subplot(gs[1, 0]) 

    # Special SPES colorbar
    else:
        # Cut the Gray colormap
        spes_cmin = 0.2 # 0.4 for Greys
        spes_cmax = 1
        # cmap = plt.get_cmap('Greys')
        cmap = plt.get_cmap('YlOrBr')
        norm = matplotlib.colors.Normalize(vmin=spes_cmin, vmax =spes_cmax)
        #generate colors from original colormap in the range equivalent to [vmin, vamx] 
        colors = cmap(np.linspace(1.-(spes_cmax-spes_cmin)/float(spes_cmax), 1, cmap.N))
        # Create a new colormap from those colors
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_Greys', colors)


        c_toplot=np.linspace(spes_cmin, spes_cmax,  c_toplot.shape[0])
        spes_sc = ax.scatter(lat_data_windowed_toplot[0,:], lat_data_windowed_toplot[1,:], c=c_toplot, alpha=plot_alpha, s=s_plot, cmap=color_map, norm=norm, edgecolors='none')
        ax.set_title('Latent Space' + title_ictal_included)

        cbar_spes = plt.colorbar(spes_sc, ax=ax, ticks=[spes_cmin, spes_cmax], orientation='vertical')
        cbar_spes.ax.set_yticklabels(['Start\nSPES', 'Stop\nSPES'])

        ax.set_ylabel("Latent Var 1")
        ax.set_xlabel("Latent Var 0")
        ax.set_aspect('equal')
        if not auto_scale_plot:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

    # Return all axes    
    xy_lims = [ax.get_xlim(), ax.get_ylim()]
    return ax, interCont_ax, seiztype_ax, time_ax, cluster_ax, xy_lims