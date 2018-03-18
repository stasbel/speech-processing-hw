import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

from tqdm import tqdm
from features.extractor import MFFeatureExctractor
from features.utils import sample_wav_by_time, chunks, in_any, interv_to_range, get_sname


class SSPNetSampler:
    """Class for loading and sampling audio data by frames for SSPNet Vocalization Corpus"""

    def __init__(self, corpus_root, frame_sec=0.5):
        self.frame_sec = frame_sec
        
        # SSPNet & core params
        self.sr = 16000
        self.duration = 11
        self.default_len = self.sr * self.duration
        self.frame_size = int(self.sr * self.frame_sec)
        self.frame_step = int(self.frame_size / 5)
        
        # Files
        self.data_dir = os.path.join(corpus_root, "data")
        self.labels = self._read_labels(os.path.join(corpus_root, "labels.txt"))

    def cook(self, wav_path):
        features = MFFeatureExctractor(self.frame_sec).extract(wav_path)
        sname, labels = self._get_labels(wav_path)
        return sname, pd.concat([features, pd.DataFrame(labels, index=features.index, dtype=int)], axis=1)

    def sample(self, naudio=None):
        fullpaths = self._get_valid_wav_paths()[:naudio]
        snames, dfs = [], []
        for sname, df in tqdm((self.cook(wav_path) for wav_path in fullpaths), total=len(fullpaths)):
            snames.append(sname)
            dfs.append(df)
        df = pd.concat(dfs, keys=snames)
        df.index.rename(['sname', 'frame'], inplace=True)
        return df

    def predicted_to_intervals(self, pred_classes, error_dist=0.1):
        if error_dist is None:
            error_dist = self.frame_sec * 1.5
        frames_to_times = [self.frame_sec * i for i, pred in enumerate(pred_classes) if pred == 1]
        intervals_g = self._intervals_gen(frames_to_times, error_dist=error_dist)
        intervals = list(intervals_g)
        return intervals
    
    def _intervals_gen(self, timestamps, error_dist=None, min_frames=None):
        if min_frames is None:
            min_frames = 10
        if error_dist is None:
            error_dist = 1.5 * self.frame_sec

        begin = 0
        length = len(timestamps)
        if length <= 1:
            return
        for i in range(1, length):
            if timestamps[i] - timestamps[i - 1] >= error_dist:
                if i - 1 - begin > min_frames:
                    yield timestamps[begin], timestamps[i - 1]
                begin = i
        if begin != length - 1 and length - 1 - begin > min_frames:
            yield timestamps[begin], timestamps[length - 1]
    
    @staticmethod
    def _read_labels(labels_path):
        def_cols = ['Sample', 'original_spk', 'gender', 'original_time']
        label_cols = ["{}_{}".format(name, ind) for ind in range(6) for name in ('type_voc', 'start_voc', 'end_voc')]
        def_cols.extend(label_cols)
        labels = pd.read_csv(labels_path, names=def_cols, engine='python', skiprows=1)
        return labels
    
    def _get_labels(self, wav_path):
        sname = get_sname(wav_path)
        sample = self.labels[self.labels.Sample == sname]

        incidents = sample.loc[:, 'type_voc_0': 'end_voc_5']
        incidents = incidents.dropna(axis=1, how='all')
        incidents = incidents.values[0]

        rate, audio = wav.read(wav_path)
        
        def interval_generator(incidents):
            for itype, start, end in chunks(incidents, 3):
                if itype == 'laughter':
                    yield start, end

        laughts = interval_generator(incidents)
        laughts = [interv_to_range(x, len(audio), self.duration) for x in laughts]
        laught_along = [1 if in_any(t, laughts) else 0 for t, _ in enumerate(audio)]
        
        def most(l):
            return int(sum(l) > len(l) / 2)
        
        is_laugh = [most(laught_along[i: i + self.frame_size]) 
                    for i in range(0, self.default_len - self.frame_size, self.frame_step)]
        
        return sname, pd.Series(is_laugh, name='laugh')
    
    def _get_valid_wav_paths(self):
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            fullpaths = [os.path.join(dirpath, fn) for fn in filenames]
            return [path for path in fullpaths if len(wav.read(path)[1]) == self.default_len]
