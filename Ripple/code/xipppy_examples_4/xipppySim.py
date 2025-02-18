from datetime import datetime
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
        return datetime.now().microsecond #microseconds to milliseconds

    def cont_raw(self, num_samps, ch_list):
        """returns num_samps for defined channels
        In reality just makes an n_samp*len(ch_list) stream of random numbers"""
        t = self.time()
        self.snippet_size = num_samps
        try:
            return next(self.stream_samps), t
        except TypeError:
            logger.error(f"self.stream_samps is not defined correctly! Val: {self.stream_samps}")
            raise BufferError("set up stream correctly!")

    def cont_lfp(self, num_samps, ch_list):
        """returns num_samps for defined channels
        In reality just makes an n_samp*len(ch_list) stream of random numbers"""
        t = self.time()
        self.snippet_size = num_samps
        try:
            return next(self.stream_samps), t
        except TypeError:
            logger.error(f"self.stream_samps is not defined correctly! Val: {self.stream_samps}")
            raise BufferError("set up stream correctly!")

    def cont_hires(self, num_samps, ch_list):
        """returns num_samps for defined channels
        In reality just makes an n_samp*len(ch_list) stream of random numbers"""
        t = self.time()
        self.snippet_size = num_samps
        try:
            return next(self.stream_samps), t
        except TypeError:
            logger.error(f"self.stream_samps is not defined correctly! Val: {self.stream_samps}")
            raise BufferError("set up stream correctly!")

    def cont_hifreq(self, num_samps, ch_list):
        """returns num_samps for defined channels
        In reality just makes an n_samp*len(ch_list) stream of random numbers"""
        t = self.time()
        self.snippet_size = num_samps
        try:
            return next(self.stream_samps), t
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
            num_samples = f.getNSamples()[0]  # Total samples in the channel
            # sample_rate = f.getSampleFrequency(channel)  # Sampling rate of the channel
            # data = f.readSignal(channel)  # Read full signal for the channel
            step_size = snippet_size - overlap
            for start in range(0, num_samples - snippet_size + 1, step_size):
                sigs = []
                for channel in self.ch_list:
                    sigs.append(f.readSignal(channel, start=start, n=snippet_size))
                ipdb.set_trace()
                yield np.concatenate(sigs)

