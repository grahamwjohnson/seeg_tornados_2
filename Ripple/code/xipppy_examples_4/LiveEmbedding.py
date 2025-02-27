"""
This is a script that takes live streams of SEEG data and runs inference
on a brain state embedding model on the SEEG data.
1. First uses digital notch filters for SEEG data
2. Feeds data into histogram eqalization pipeline
3. Runs hist eq data through GPU and performs inference on model
4. Generates an embedding then feeds embedding of last 5s of data into pacmap-> generates an embedding
5. plots this embedding relative to a preplotted 100 point plot 

Created on Janurary 2025

@author: richard
@author: ghassan makhoul, extended original code and adapted for tranformers
"""
SIM_MODE = True
N_CH =129
import ipdb
import time
from datetime import datetime, timedelta
import sys
import re
import yaml
sys.path.append("/home/ghassanmakhoul/Documents/Tornadoes_v1/")
sys.path.append("/home/ghassanmakhoul/Documents/Tornadoes_v1/models")
sys.path.append("/home/ghassanmakhoul/Documents/Tornadoes_v1/train")
sys.path.append('/home/ghassanmakhoul/Documents/Tornadoes_v1/preprocess')
from utilities.utils_functions import apply_wholeband_filter
from scipy.signal import resample
import gc
from time import sleep
import loguru
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#use xipppySim if not connected to ripple

import sys
from train_model import *
import matplotlib
from loguru import logger
from pacmap import pacmap
import seaborn as sns

# Explicitly set the backend to TkAgg, avoid conflict with PyQT6
matplotlib.use('TkAgg')
from matplotlib.colors import LinearSegmentedColormap

if SIM_MODE:
    from xipppySim import RippleSim
    snippet_size = 4*512 #number of seconds * sampling_freqdf
    N_CH_EDF = 148 # this mayh be different than them model was trained on because of the addition of Micro+MAcro
                # We often ignore the micro channels and thus a discrepancy
    stream_edf = "/home/ghassanmakhoul/Documents/data/Spat113/Spat113_02012025_07014401_clabel.EDF"
    xp = RippleSim(stream_edf, snippet_size, [i for i in range(N_CH_EDF)])
    #set up stream here
else:
    import xipppy as xp

class LiveStreamBrainStateEmbedding:
    """
    This class sets up and displays a livestream of acquired data and its live periodogram.
    """

    def __init__(self,subject, model_file, hist_eq_pkl, pac_prefix, config_f, display_s=5, window_ms=5000, stream_ch=[a for a in range(128)], stream_ty='hifreq', gpu_id=0, **kwargs):
        """
        Initialization function for setting up the LiveStreamPeriodogram class.

        Parameters:
            subject : subject name (should follow 'pat' convention) Important for streaming and saving of data
            model_file : location of the saved model weights for running inference, TODO: update to reflect how /where model loaded
            pre_proc_pkl : location of a pickle file containing the histogram eqalization code
            display_s: Number of seconds of data to display on live plots
            window_ms: Number of milliseconds in FFT sliding window
            stream_ch: Channel number to record from
            stream_ty: Type of data stream to record from ('raw', 'lfp', 'hires', 'hifreq')
        """
        self.subject = subject
        self.display_s = display_s
        self.window_ms = window_ms
        self.stream_ch = stream_ch
        self.stream_ty = stream_ty
        if SIM_MODE:
            self.ripple_freq = 512
            self.n_stream_ch = 148
        else:
            self.ripple_freq = 7500 # TODO add cases for streams later. for now hifreq
            self.n_stream_ch = len(self.stream_ch)
        self.event_df = kwargs['event_df']
        self.bip_df_path = kwargs['bip_path']
        self.bip_df = pd.read_csv(self.bip_df_path)
        self.model_sampling_freq = 512 #magic natus number for now
        self.pac_prefix = pac_prefix
        self.model_file = model_file
        self.pred_batch = []
        self.pacmap_win_size = 60 # size in seconds Hard coded for now, TODO review and make modular for future generation
        self.transformed_latent_space = []
        self.curr_prediction_count = 0
        self.model_kwargs = dict()
        self.gpu_id = gpu_id
        self.config_f = config_f
        # Setup frequency range (x-axis limits) and decibel range (y-axis limits) for periodogram
        self.f_range = [0, 2000]
        self.db_range = [-50, 100]

        # Initialize connection to Neural Interface Processor
        self.connect_to_processor()

        # Pause for display duration to fill signal buffer
        sleep(self.display_s)

        # Load histogram eq pkl for eqalizing data
        self.hist_eq = self.setup_hist_equalize(hist_eq_pkl)
        #Load model for inference 
        self.model_kwargs = {}
        self.model = self.setup_inference()
        # set up empty tensor to track embeddings 
        self.curr_embeddings = torch.zeros(int(display_s), self.model_kwargs['transformer_seq_length'],self.model_kwargs['latent_dim'])
        # Setup parameters for performing live pacmap calculation
        self.sampling_rate, self.window_samp, self.n_sig, self.t_sig, = self.setup_stream()
        self.PACMAP = None
        self.setup_pacmap_settings(**self.model_kwargs) #should set pacmap
        self.projections = []

        # Setup plots for live signal and live periodogram subplots
        self.colors = None 
        self.mymap = None
        # get the colors from the color map
        self.my_colors = []
        self.pt_colors = []
        self.live_plot, self.ax_pre, self.ax_live, \
         self.h_pre, self.h_live, self.ax_raw, self.ax_clean, self.ax_histeq= self.setup_plots()
        self.h_raw, self.h_clean, self.h_histeq = None, None, None

        self.raw_sig = np.array([])
        self.clean_sig = np.array([])
        self.eq_sig = np.array([])

    def connect_to_processor(self):
        """
        Connect to the Neural Interface Processor. If UDP connection fails, 
        attempt TCP connection.
        """
        # Close Connection
        xp._close()
        sleep(0.001)

        # Connect to processor (Try UDP then TCP).
        try:
            xp._open()
        except:
            try:
                xp._open(use_tcp=True)
            except:
                print("Failed to connect to processor.")
            else:
                print("Connected over TCP")
        else:
            print("Connected over UDP")

        sleep(0.001)

    def clean_signal(self, reshaped_stream):
        """Apply preprocessing pipeline to reshaped
        input stream
        1. subsample to 512 Hz
        3. Bipole montage
        2. preprocessing code: filter
        

        Args:
            reshaped_inp_stream (_type_): filtered and cleaned data
        """
        def resample_multivariate_time_series(data, F_in, F_out, inflate=True):
            """
            Resample a multivariate time series from F_in to F_out.
            
            Parameters:
                data (numpy.ndarray): The input multivariate time series with shape (num_samples, num_features).
                F_in (float): The input sampling frequency.
                F_out (float): The desired output sampling frequency.
            Return
            numpy.ndarray: The resampled multivariate time series with shape (new_num_samples, num_features).
            """
            # Calculate the ratio of the output to input frequencies
            ratio = F_out / F_in
            # Calculate the new number of samples
            num_samples = data.shape[1]
            new_num_samples = int(np.round(num_samples * ratio))
            # Resample each feature (column) in the time series
            resampled_data = np.zeros((data.shape[0], new_num_samples))
            for i in range(data.shape[0]):
                resampled_data[i, :] = resample(data[i, :], new_num_samples, axis=0)
            if inflate:
                resampled_data = np.append(resampled_data, resampled_data, axis=0)
                resampled_data = np.append(resampled_data, resampled_data, axis=0)
                resampled_data = np.append(resampled_data, resampled_data, axis=0)
            return resampled_data
        logger.info(f"Subsampling to {self.model_sampling_freq}")
        if not SIM_MODE:
            #Sim mode will return 512Hz data
            reshaped_stream = resample_multivariate_time_series(reshaped_stream, self.ripple_freq, self.model_sampling_freq)
        logger.info("Assigning bipolar montage")
        # Check that bip names match based on already created montage from previous files
        
        mont_idxs = self.bip_df.mont_idxs.values
        bip_names = self.bip_df.bip_names.values

        # If we made it this far, then the montage aligns across EDF files
        # Re-assign data to bipolar montage (i.e. subtraction: A-B)
        
        new_reshaped_stream = np.empty([len(bip_names),reshaped_stream.shape[1]], dtype=np.float16)
        for i in range(0, len(bip_names)):
            a, b = mont_idxs[i].strip("[]").replace(' ', '').split(',')
            new_reshaped_stream[i,:] = reshaped_stream[int(a),:] - reshaped_stream[int(b),:]
        reshaped_stream = new_reshaped_stream
        del new_reshaped_stream
        gc.collect()
        reshaped_filt = apply_wholeband_filter(reshaped_stream, self.model_sampling_freq)
        return reshaped_filt


    def setup_hist_equalize(self, pre_proc_pkl:str):
        """set up histogram eqalization runner

        Args:
            pre_proc_pkl (str): Location of the pickle file for the histogram eqalization code
        """
        import joblib
        with open(pre_proc_pkl, "rb") as f:
            linear_interp_by_ch = joblib.load(f)
        self.hist_eq = linear_interp_by_ch
        
    
    def hist_equalize(self, filt_data):
        scaled_filt_data = np.zeros(filt_data.shape)
        n_ch = filt_data.shape[0]
        for ch_idx in range(0,n_ch):
            scaled_filt_data[ch_idx,:] = self.hist_eq[ch_idx](filt_data[ch_idx,:])
        scaled_filt_data = scaled_filt_data.clip(-1,1)
        return scaled_filt_data
        
    def setup_inference(self):
        """loads the pytorch model to perform live inference

        Args:
            model_loc (str): model location
        """

        #TODO: implement make modular in future
        assert os.path.exists(self.model_file), f"Need to pass valid model file location. Got: {self.model_file}"
        assert self.config_f != '' and os.path.exists(self.config_f), "Pass in valide config file"
        with open(self.config_f, 'r') as f : kwargs = yaml.load(f, Loader=yaml.FullLoader)
        kwargs = utils_functions.exec_kwargs(kwargs) # Execute the arithmatic build into kwargs and reassign kwargs
        world_size = torch.cuda.device_count()
        #NOTE: very important that you feed in
        # the same number type of kwargs to inference
        self.model_kwargs = kwargs 
        if world_size == 1:
            self.gpu_id = 0
        else:
            self.gpu_id = 0
        vae = VAE(gpu_id=self.gpu_id, **kwargs)
        vae = vae.to(self.gpu_id)
        vae.eval()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.gpu_id}
        vae_state_dict_prev = torch.load(self.model_file, map_location=map_location)
        
        
        vae.load_state_dict(vae_state_dict_prev,strict=False) 

        return vae
    

    def setup_stream(self):
        """
        This function will enable the selected stream type for the Neural Interface Processor
        and calculate the necessary Pacmap settings

        Returns:
            sampling_rate: Sampling frequency in Hz of the selected datastream
            window_samp: Number of samples in the FFT window
            N: Most efficient size of the N-Point FFT
            n_sig: Number of signal points in the display window
            t_sig: Time series of the signal points in the display window
            f_psd_total: Frequency series of the FFT calculation
            f_ind: Frequency indices for the selected frequency range
            f_psd: Frequency series for Periodogram display
        """
        if SIM_MODE:
            sampling_rate = 512
        elif self.stream_ty == 'raw':
            sampling_rate = 30000
        elif self.stream_ty == 'lfp':
            sampling_rate = 1000
        elif self.stream_ty == 'hi-res':
            sampling_rate = 2000
        elif self.stream_ty == 'hifreq':
            sampling_rate = 7500
        else:
            sys.exit(f'{self.stream_ty} is an invalid stream type.\n')
        #load pacmap in 
        
        # Enable stream type on Neural Interface Processor if not enabled
        for ch in self.stream_ch:
            if not xp.signal(ch, self.stream_ty):
                xp.signal_set(ch, self.stream_ty, True)

        window_samp = int(np.floor(self.window_ms * sampling_rate / 1e3))  # FFT window size in samples
        
        n_sig = int(sampling_rate / 1e3 * round(self.display_s * 1e3))  # Number of signal data points in display window
        t_sig = np.linspace(-self.display_s, 0, n_sig)  # Time for Signal
        

        return sampling_rate, window_samp, n_sig, t_sig

    def setup_pacmap_settings(self, num_iters=(150,150,150), **kwargs):
            #NOTE: why am I loading the pacmap and setting up the streaming settings all at once?
            self.PACMAP = pacmap.load(self.pac_prefix)
            self.PACMAP.num_iters = num_iters


    def setup_plots(self):
        """
        This function sets up the subplots for displaying a live pacmap and a previous pacmap from BSE.

        Returns:
            live_plot: Figure object for the plots
            ax_pre Axes object for the pretrained pacmap projection
            ax_live: Axes object for the live rendered plot
            h_pre: Line2D object for the signal plot
            h_live: Line2D object for the periodogram plot data
        """
        # plot hour of data ahead of time and buffer period
        # then update only the last hour 
        # can PacMAP be meaningful enough to see us go into the preictal funnel
        # perfect world: ghoast of last hour, so the most recent points are darkest
        # flow of data: filter -> hist_eq -> run through model -> PacMAP model -> update in FIFO manner with last 10 minutes-hour as 
        # static plot
        # saving all data to a cold storage archive 
        # code should update buffer and a masterfile
        # don't need to replot the same background points 
        # generate a numpy random array (-5 +5 on all axes, and incorporate time stamps for seizures appropriately)
        lim_file = self.pac_prefix.replace("PaCMAP","") + 'xy_lims.pkl'
        with open(lim_file, 'rb') as f: lims = pickle.load(f)
        gs = gridspec.GridSpec(4, 2, height_ratios=[6, 1, 1, 1])  # Top row is twice as tall!


        xlims, ylims = lims
        live_fig = plt.figure(figsize=(8,11))
        x = np.random.random(10)
        y = np.sin(x)
        # Pretrained pacmap values here 
        # TODO find out the dimensions of the PACMAP
        ax_pre = live_fig.add_subplot(gs[0,0])
        ax_pre.set_aspect('equal', adjustable='box')
        h_pre = ax_pre.scatter(x,y)  # Add colors
        ax_pre.set_xlim(*xlims)
        ax_pre.set_ylim(*ylims)
        ax_pre.set_title(f'Pretrained   Embedding: {self.subject}')
        ax_pre.set_xlabel('Dimension 1')
        ax_pre.set_ylabel('Dimension 2')
        # Add pacmap
        ax_live =live_fig.add_subplot(gs[0,1])
        ax_live.set_aspect('equal', adjustable='box')
        h_live = ax_live.scatter(x,y)  # Live PacMAP embeddings
        ax_live.set_xlim(*xlims)
        ax_live.set_ylim(*ylims)
        ax_live.set_title(f'Live Prediction: {self.subject}')
        ax_live.set_xlabel('Dimension 1')
        ax_live.set_ylabel('Dimension 2')

        ##Set up colors for PaCMAP scatter
        self.colors = np.r_[np.linspace(0.1, 1, 5), np.linspace(0.1, 1, 5)] 
        self.mymap = plt.get_cmap("Reds")
        # get the colors from the color map
        self.my_colors = self.mymap(self.colors)

        #setup SEEG plots
        ax_raw = live_fig.add_subplot(gs[1,:])
        ax_clean=live_fig.add_subplot(gs[2,:])
        ax_histeq = live_fig.add_subplot(gs[3,:])
        plt.subplots_adjust(hspace=0.55)
        plt.suptitle("Live Brain State Embedding", fontweight='bold')

        return live_fig, ax_pre, ax_live, h_pre, h_live, ax_raw, ax_clean, ax_histeq
    
    def reshape_stream(self, stream_data):
        """Reshape stream data from a [n_samp*n_ch,1] array to a 
        an array of shape [n_ch, n_ch]

        Args:
            stream_data (np.array): Raw stream
        """
        stream_data = np.array(stream_data)
        return stream_data.reshape(self.n_stream_ch,-1)
    
    #@logger.catch
    def batch_inference(self, inp_stream):
        """Batches data to feed into the self attentional
        VAE forward pass
        data will enter as n_ch x n_times points as a stream.
        Remember that we are interested in using 8 tokens at a time
        to predict 2 samples worth of data.

        Parameters:
            inp_steam (numpy.ndarray): n_channels x n_sample input stream

        256 samples x n_ch x 2 samples
        
        NOTE: these sizes are all determined by model settings
        that can be found in config.yml. Thus this introduces a possible
        point of failure if the config file does not reflect the most
        up to date model params
        """
        with torch.no_grad(): # Does not allocate for autograd/build computational graph.
            n_ch, t_samps = inp_stream.shape
            params = self.model_kwargs
            n_ch_model = params['padded_channels']
            num_samples_in_forward = params['transformer_seq_length']* params['autoencode_samples']
            num_windows_in_file = inp_stream.shape[-1] /num_samples_in_forward
            assert(num_windows_in_file %1) == 0, "Error in dividing up input steam into batches!"
            num_windows_in_file = int(num_windows_in_file)
            num_samples_in_forward = int(num_samples_in_forward)

            # reshape data to create a batch_size x n_ch x t_samps tensor
            # currently using naive method just because i know it works
            data = torch.zeros(num_windows_in_file, n_ch_model, num_samples_in_forward)
            inp_stream = torch.from_numpy(inp_stream)
            inp_stream = inp_stream.type(data.dtype) # cast input stream to same type as data 
            s, e = 0, num_samples_in_forward
            ch_idx = np.arange(n_ch)
            for w in range(num_windows_in_file):

                data[w,ch_idx,:] = inp_stream[:,s:e]
                s = e
                e = (w+1)*num_samples_in_forward + num_samples_in_forward
            # ipdb.set_trace()
            # collect sequential embeddings for transformer to pseudo-batch
            # In this case num_windows_in_file will be used as the batch dimension
            # double check implementation 
            for w in range(1):
                x = torch.zeros(num_windows_in_file, params['transformer_seq_length'], n_ch_model,params['autoencode_samples']).to(self.gpu_id)
                start_idx = w*num_samples_in_forward
                for embedding_idx in range(0,params['transformer_seq_length']):
                    end_idx = start_idx + params['autoencode_samples']*embedding_idx + params['autoencode_samples']
                    x[:,embedding_idx,:,:] = data[:,:,end_idx-params['autoencode_samples']: end_idx]
                mean,_,_,_ = self.model.forward(x, reverse=False)

                mean = mean.cpu().numpy()
                # mean = torch.stack(\
                #     torch.split(mean_batched, \
                #         params['transformer_seq_length'] - params['num_encode_concat_transformer_tokens'], \
                #         dim=0), dim=0)
                # no longer have to handle, thanks to Graham-y:)
                self.update_prediction(mean)
            logger.info(f"Ran inference on {num_windows_in_file} windows ")
        gc.collect()
        return


    def query_model(self, data: np.array):
        """Queries a model to run inference on, 
        should take advantage of GPU and pretrained model
        
        ASSUMES that the model has been instantiated/loaded

        Args:
            data (np.array): N_ch x M_samples array of live ecog data
        """
        
        if self.model == None: raise ValueError("Cannot run inference if model has not been loaded")
        _, hash_channel_order = utils_functions.hash_to_vector(
                            input_string=self.subject, 
                            num_channels=data.shape[0], 
                            latent_dim=self.model.latent_dim, 
                            modifier=0)
        data = data[hash_channel_order,:]
        # chunk into time points of 2
        # make a batch sizec of 5 with 256 sequential subwindows, each 2 datapoints wide with n_ch
        # within forward pass, model is gonna shuffle axes like crazy
        # should get back n_ch x 1024 
        # mean across the 248 mean dimension 
        # Inpt dim : [batch_size x transformer_seq_length, pat_ch x auto_encoder_samps]
        # 
        
        self.batch_inference(data)

    def calc_seizure_risk(self, ts, sim_mode=SIM_MODE):
        """Calculates seizure risk and reports a live score for bar chart"""
        def conv_datetime(row):
            """converts our event csv format to datetime obj"""
            format_string = "%m:%d:%Y %H:%M:%S"
            day = row['Date (MM:DD:YYYY)'].values[0]
            time_of_day = row['Onset String (HH:MM:SS)'].values[0]
            date_str = day.values[0] + " " + time.values[0]
            return  datetime.strptime(date_str, format_string)


        if sim_mode:
            #shouldn't calc each time but for now okay            
            prev_events = self.event_df.apply(conv_datetime, axis=0)
            time_diff = prev_events - ts

            any_below_2_hours = (time_diff.dt.total_seconds() < 2 * 3600).any()
            if any_below_2_hours:
                logger.warning("WITHIN 2 hours of a seizure")
                
                return    

            any_below_4_hours = (time_diff.dt.total_seconds() < 4 * 3600).any()
            if any_below_4_hours:
                logger.warning("WITHIN 4 hours of a seizure")
                return
            return
            #calculate based on time to nearest seizure


    def update_time_now(self) -> bool:
        """Returns True/False if enough 
        embeddings have been collected to trigger a PaCMAP update
        """
        return self.curr_prediction_count >0 and self.curr_prediction_count %self.pacmap_win_size == 0
    
    def update_prediction(self, mean_batched):
        """Updates a running buffer of the current prediction
        If we have reached the cutoff for a prediction batch_size
        then update pacmap embeddings"""
        self.curr_prediction_count += self.display_s #THIS assumes that seconds of display window produce seconds of batch
        if self.update_time_now():
            #Refresh embeddings every 60s (determined by pacmap win_size)
            self.update_embeddings()
            #resets the prediction batch
            # in future save out?
            self.pred_batch = []
            
            
        mean_seq = np.mean(mean_batched,axis=1) #get mean embedding per-pseudobatch
        mean_of_means = np.mean(mean_seq, axis=0) #
        mu = mean_of_means
        self.pred_batch.append(mu.reshape(1,-1))

    
    def update_embeddings(self):
        """Once enough embeddings have been generated to 
        sufficiently run pacmap, update the current embeddings
        """
        mean_proj = np.mean(np.concatenate(self.pred_batch), axis=0).reshape(1,-1)
        #since we are passing one sample at a time, just pull out first set of coords
        logger.info(f"Shape of updating projection: {mean_proj.shape}")
        transformed_latent_space = self.PACMAP.transform(mean_proj)[0]
        if len(self.projections) == 0:
            self.projections = transformed_latent_space 
        else:
            self.projections = np.vstack([self.projections, transformed_latent_space])
        logger.success(f"Updated embeddings for time:{self.curr_prediction_count-self.pacmap_win_size}s to {self.curr_prediction_count}s")
        return
    
    def update_plots(self,i):
            """Subroutine that updates all plots, live_trajectories, example seeg_raw, seeg_clean, seeg_histeq
            #if we have accumulated 60s worth of embedding, then pacmap 
            # has enough information to generate meaningful low dim representation
            
            # Plot updated data and rescale axes as needed
            # assume embeddings have shape N_samp x 2_dim

            """
            assert  len(self.raw_sig) != 0 and len(self.clean_sig) != 0 and len(self.eq_sig) != 0, "Signals have not been added to plot vars"
            #update live pacmap
            ipdb.set_trace()
            self.pt_colors.append(self.my_colors[i%5])
            self.h_live.set_offsets(self.projections )
            self.h_live.set_facecolor(self.pt_colors)
            self.ax_live.autoscale_view()
            # self.ax_sig.relim()
            # Line plots, typically gonna be lineplot
            self.ax_clean, self.ax_histeq
            t_raw = np.arange(self.raw_sig.shape[1])/self.ripple_freq
            t_hist_eq = t_clean = np.arange(self.clean_sig.shape[1])/self.model_sampling_freq
            ch = np.random.randint(0, self.bip_df.shape[0])
            ch_names = self.bip_df.bip_names.values[ch]

            if self.h_raw == None:
                self.h_raw = self.ax_raw.plot(t_raw, self.raw_sig[ch,:])
                self.ax_raw.set_title(f'Raw SEEG for  {self.subject} at Ch {ch_names}')
                self.ax_raw.set_ylabel('uV')
                self.ax_raw.set_xlabel('Time (s)')
                

                self.h_clean = self.ax_clean.plot(t_clean, self.clean_sig[ch,:])
                self.ax_clean.set_title(f'Clean SEEG for  {self.subject} at Ch {ch_names}')
                self.ax_clean.set_ylabel('uV ')
                self.ax_clean.set_xlabel('Time (s)')
                self.h_histeq = self.ax_histeq.plot(t_hist_eq, self.eq_sig[ch,:])

                self.ax_histeq.set_title(f'Histeq SEEG for  {self.subject} at Ch {ch_names}')
                self.ax_histeq.set_ylabel('uV (eqed)')
                self.ax_histeq.set_xlabel('Time (s)')
            else:
                self.h_raw[0].set_ydata(self.raw_sig[ch,:])
                self.h_clean[0].set_ydata(self.clean_sig[ch,:])
                self.h_histeq[0].set_ydata(self.eq_sig[ch,:])
            self.ax_pre.set_ylim((-28,28))
    
            self.ax_live.set_ylim((-28,28))
            plt.tight_layout()
            return


         

    def live_data_loop(self):
        """
        This function performs the live collection of data and updates the plot
        in a loop while the figure is open.
        """
        # Get current Neural Interface Processor time
        t0 = xp.time()

        # While plot is open, continue live plotting data
        prediction_num = self.curr_prediction_count
        i = 0
        while True:
            t1 = xp.time()
           
            # logger.info(f"Loop {i}: with {t1},{t0} condition {(t1 - t0) >= (3e4 / 30)}")

            # If time since last loop is greater than 1/30 seconds, update loop
            # (Frame rate is capped at 30 FPS)
            if (t1 - t0) >= 10:#(3e4 / 30)*self.display_s: #has 1 second passed?
                t0 = t1
                
                # Data collection based on stream type
                if self.stream_ty == 'raw':
                    x_sig, ts = xp.cont_raw(round(self.display_s * self.sampling_rate), self.stream_ch)
                elif self.stream_ty == 'lfp':
                    x_sig, ts = xp.cont_lfp(round(self.display_s * self.sampling_rate), self.stream_ch)
                elif self.stream_ty == 'hi-res':
                    x_sig, ts = xp.cont_hires(round(self.display_s * self.sampling_rate), self.stream_ch)
                elif self.stream_ty == 'hifreq': 
                    x_sig, ts = xp.cont_hifreq(round(self.display_s * self.sampling_rate), self.stream_ch)
                
                # Calculation of FFT and periodogram data
                #TODO double check reshape order check element to element
                sig_reshape = self.reshape_stream(x_sig)
                # Filter 60hz line noise and resample to 512Hz
                # do the exact preprocessing filtering
                clean_sig = self.clean_signal(sig_reshape)

                # Can apply eq parameters, open pickle and use logic for the right channel
                # The setup goes here 
                
                eq_sig = self.hist_equalize(clean_sig)
                self.query_model(eq_sig)
                self.calc_seizure_risk(ts)
                # Make sure to load random model and feed in dummy data 
                # TODO - load pangolin, initialize, load weights, torch.rand(with correct shape)
                #           - pull out the mu, check the pacmap forloop where it calles pacmap then fine tunes
                #           - use that inference code run_epoch in training object
                #           - if get_latent_only (diff forward pass)
                #           -  forward passes a second of data, get the latent prediction at 2-samples of stride (248 predictions)
                #           - then take the average of those predictions, and get average mu
                #           - so I'll feed in a second and get 1024 dim x 1_sec of data
                #           - loop time needs to be faster than a second, or batch into 5 seconds, and just run over 5s and get 5 latent vectors
                #           - if loop time takes 3seconds then do a 5s stride so we get time to run
                if self.update_time_now():
                    self.raw_sig = sig_reshape
                    self.clean_sig = clean_sig
                    self.eq_sig = eq_sig
                
                    self.update_plots(i)
                    i += 1
                # self.h_psd.set_offsets(np.c_[x_sig, x_sig])
                    plt.pause(0.05)  # Small pause to update the plot
                    plt.savefig("tst.png")
  


if __name__ == '__main__':

    n_ch = [a for a in range(N_CH)] # num of channels for Spat 113
    model_file = "/home/ghassanmakhoul/Documents/trained_models/pangolin_finetune/Epoch_297/core_checkpoints/checkpoint_epoch297_vae.pt"
    hist_eq_pkl = '/home/ghassanmakhoul/Documents/results/Bipole_datasets/By_Channel_Scale/HistEqualScale/data_eqalized_to_first_24_hours/wholeband/DurStr_1024s896s_epoched_datasets/Spat113/scaled_data_epochs/metadata/scaling_metadata/linear_interpolations_by_channel.pkl'
    pac_prefix = '/home/ghassanmakhoul/Documents/trained_models/pangolin_finetune/Epoch_297/pacmap_generation/epoch298_PaCMAP'
    config_f = '/home/ghassanmakhoul/Documents/Tornadoes_v1/config_live.yml'
    event_df = pd.read_csv("/home/ghassanmakhoul/Documents/Tornadoes_v1/all_time_data_01092023_112957.csv")
    bip_path = "/home/ghassanmakhoul/Documents/data/Resamp_512_Hz/bipole_filtered_wholeband_unscaled_big_pickles/Spat113/metadata/Spat113_bipolar_montage_names_and_indexes_from_rawEDF.csv"
    pat_id = 'Spat113'
    pat_event_df = event_df['Pat ID'] = pat_id
    processor = LiveStreamBrainStateEmbedding(pat_id,model_file, hist_eq_pkl, pac_prefix, config_f,  display_s=4, stream_ch=n_ch, event_df=pat_event_df, bip_path=bip_path)
    processor.live_data_loop()
    plt.show()
    xp._close()