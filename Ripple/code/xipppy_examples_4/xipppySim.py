import time
import datetime
import pyedflib
import numpy as np
from loguru import logger
import ipdb

STREAM_ENABLED = False
class RippleSim:
    def __init__(self, edf_file, snippet_size, ch_list):
        self.edf_file = edf_file
        self.snippet_size = snippet_size
        self.ch_list = ch_list
        self.stream_samps = None
        self.start_time = int(time.time()*1000) #start time in milliseconds
        self.edf_start_time = None
        self.edf_end_time = None
        self.fs = None
    
    def _open(self, use_tcp=False):
        if use_tcp:
            logger.info("Simulating over TCP")
        logger.info("Xipppy Simulator active")
        #No overlap at first
        self.stream_samps = self.stream_edf(overlap=0)

    def _close(self):
        logger.info("Xipppy simulator closed")

    def signal(self, ch, stream_ty):
        logger.info(f"Status of {ch} for {stream_ty} streams: {STREAM_ENABLED}")
        return STREAM_ENABLED

    def signal_set(self, ch, stream_ty, enable):
        global STREAM_ENABLED 
        STREAM_ENABLED= True
        logger.success(f"{ch} channels enabled with {stream_ty} samples/sec")

    def time(self):
        return int(time.time()*1000) - self.start_time #microseconds to milliseconds

    def cont_raw(self, num_samps, ch_list):
        """returns num_samps for defined channels
        In reality just makes an n_samp*len(ch_list) stream of random numbers"""
        t = self.time()
        self.snippet_size = num_samps
        try:
            return next(self.stream_samps)
        except TypeError:
            logger.error(f"self.stream_samps is not defined correctly! Val: {self.stream_samps}")
            raise BufferError("set up stream correctly!")

    def cont_lfp(self, num_samps, ch_list):
        """returns num_samps for defined channels
        In reality just makes an n_samp*len(ch_list) stream of random numbers"""
        t = self.time()
        self.snippet_size = num_samps
        try:
            return next(self.stream_samps)
        except TypeError:
            logger.error(f"self.stream_samps is not defined correctly! Val: {self.stream_samps}")
            raise BufferError("set up stream correctly!")

    def cont_hires(self, num_samps, ch_list):
        """returns num_samps for defined channels
        In reality just makes an n_samp*len(ch_list) stream of random numbers"""
        t = self.time()
        self.snippet_size = num_samps
        try:
            return next(self.stream_samps)
        except TypeError:
            logger.error(f"self.stream_samps is not defined correctly! Val: {self.stream_samps}")
            raise BufferError("set up stream correctly!")

    def cont_hifreq(self, num_samps, ch_list):
        """returns num_samps for defined channels
        In reality just makes an n_samp*len(ch_list) stream of random numbers"""
        t = self.time()
        self.snippet_size = num_samps
        try:
            return next(self.stream_samps)
        except TypeError:
            logger.error(f"self.stream_samps is not defined correctly! Val: {self.stream_samps}")
            raise BufferError("set up stream correctly!")

    def stream_edf(self, overlap=0):
        """
        Generator that yields successive snippets of EEG data from an EDF file.
        
        Parameters:
            edf_file (str): Path to the EDF file.
            snippet_size (int): Number of samples per snippet.
            channel (int): Channel index to extract data from.
            overlap (int): Number of overlapping samples between snippets (default: 0).
        
        Yields:
            numpy.ndarray: A snippet of EEG data.
        """
        snippet_size = self.snippet_size
        start = 0
        with pyedflib.EdfReader(self.edf_file) as f:
            num_samples = f.getNSamples()[0] 
            self.edf_start_time = f.getStartdatetime() # Total samples in the channel
            duration = f.getFileDuration()
            self.edf_end_time = self.edf_start_time + datetime.timedelta(seconds=duration)
            ipdb.set_trace()
            # sample_rate = f.getSampleFrequency(channel)  # Sampling rate of the channel
            # data = f.readSignal(channel)  # Read full signal for the channel
            step_size = snippet_size - overlap
            self.fs = f.getSampleFrequencies()[0] # assumes that all sampling freq are the same

            for start in range(0, num_samples - snippet_size + 1, step_size):
                sigs = []
                curr_time = self.edf_start_time + datetime.timedelta(seconds=start/self.fs)
                for channel in self.ch_list:
                    sigs.append(f.readSignal(channel, start=start, n=snippet_size))
                yield np.concatenate(sigs),curr_time

"""
seizure times 
Seizure,Spat113,1,01:31:2025,02:35:39,02:37:35,FIAS,0,2025-01-31 02:35:39.000000,2025-01-25 02:37:35.000000,,,,
Seizure,Spat113,2B,02:01:2025,19:50:06,19:51:28,Focal unknown awareness,0,2025-02-01 19:50:06.0000,2025-02-01 19:51:28.0000,,,,
Seizure,Spat113,3B,02:02:2025,13:49:39,13:51:20,Focal unknown awareness,0,2025-02-02 13:49:39.0000,2025-02-02 13:51:20.0000,,,,
Seizure,Spat113,4b,02:02:2025,18:09:40,18:11:14,Focal unknown awareness,0,2025-02-02 18:09:40.0000,2025-02-02 18:11:14.0000,,,,
Seizure,Spat113,5b,02:03:2025,09:50:22,09:52:03,Focal unknown awareness,0,2025-02-03 09:50:22,2025-02-03 09:52:03,,,,
Seizure,Spat113,6b,02:03:2025,13:33:29,13:35:08,Focal unknown awareness,0,2025-02-03 13:33:29, 2025-02-03 13:35:08,,,,
"""