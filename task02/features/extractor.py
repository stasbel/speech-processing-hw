import librosa
import numpy as np
import pandas as pd


class MFFeatureExctractor:
    def __init__(self, frame_sec):
        self.frame_sec = frame_sec
    
    def extract(self, wav_path):
        y, sr = librosa.load(wav_path, dtype=float)
        frame_size = int(sr * self.frame_sec)
        frame_step = int(frame_size / 5)
        
        features = []
        
        # MFCC
        mfcc = [] 
        for i in range(0, len(y) - frame_size, frame_step):
            frame_y = y[i: i + frame_step]
            mfcc.append(np.mean(librosa.feature.mfcc(y=frame_y, sr=sr), axis=1))
        mfcc = np.vstack(mfcc)
        features.append(pd.DataFrame(mfcc, 
                                     index=list(range(mfcc.shape[0])), 
                                     columns=[f'mfcc_{i}' for i in range(mfcc.shape[1])]))
        
        # FBANK
        fbank = [] 
        for i in range(0, len(y) - frame_size, frame_step):
            frame_y = y[i: i + frame_step]
            fbank.append(np.mean(librosa.feature.melspectrogram(y=frame_y, sr=sr), axis=1))
        fbank = np.vstack(fbank)
        features.append(pd.DataFrame(fbank, 
                                     index=list(range(fbank.shape[0])), 
                                     columns=[f'fbank_{i}' for i in range(fbank.shape[1])]))
        
        return pd.concat(features, axis=1)
