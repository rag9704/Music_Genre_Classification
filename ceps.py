import os
import glob
import sys

import numpy as np
import scipy
import scipy.io.wavfile

from librosa.feature import mfcc
import librosa
#from python_speech_features import mfcc

from utils import GENRE_DIR


def write_ceps(ceps, fn):
    """
    Write the MFCC to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print("Written %s"%data_fn)


def create_ceps(fn):
    #sample_rate, X = scipy.io.wavfile.read(fn)
    X,sample_rate = librosa.load(fn)
    #print(sample_rate)
    #print(X)
    
    
    mfccs=mfcc(X,sr=sample_rate)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    print(mfccs.shape)
    #ceps, mspec, spec = mfcc(X,sr=sample_rate)
    #write_ceps(ceps, fn)


def read_ceps(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            num_ceps = len(ceps)
            X.append(
                np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0))
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    os.chdir(GENRE_DIR)
    for fn in glob.glob(r"C:/Users/Rag9704/Pictures/genres/**/*.wav", recursive=True):
        create_ceps(fn)
