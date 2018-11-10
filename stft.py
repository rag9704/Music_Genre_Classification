import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas
import librosa
import sklearn
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import wave
import sys
import librosa.display as lib
import librosa.core as cr

x_hiphop,sampling_rate =librosa.load(r"C:\Users\Rag9704\Pictures\genres\hiphop\hiphop1.wav")
x_classical,sampling_rate =librosa.load(r"C:\Users\Rag9704\Pictures\genres\classical\classical1.wav")
hop_size=512
"""
fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
C = librosa.cqt(x_hiphop, sr=sampling_rate,hop_length=hop_size,fmin=librosa.note_to_hz('C2'))
logC = librosa.amplitude_to_db(C)
plt.title('Hip-Hop')
lib.specshow(logC, sr=sampling_rate, x_axis='time', y_axis='cqt_note',cmap='coolwarm',)
plt.colorbar(format='%+2.0f dB')

plt.subplot(2,1,2)
C2 = librosa.cqt(x_classical, sr=sampling_rate,hop_length=hop_size,fmin=librosa.note_to_hz('C2'))
logC2 = librosa.amplitude_to_db(C2)
plt.title('Classical')
lib.specshow(logC2, sr=sampling_rate, x_axis='time', y_axis='cqt_note',cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
"""

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
C = librosa.feature.chroma_cqt(x_hiphop, sr=sampling_rate,hop_length=hop_size,fmin=librosa.note_to_hz('C2'))
plt.title('Hip-Hop')
lib.specshow(C, sr=sampling_rate, x_axis='time', y_axis='chroma',cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2,1,2)
C2 = librosa.feature.chroma_cqt(x_classical, sr=sampling_rate,fmin=librosa.note_to_hz('C2'))
plt.title('Classical')
lib.specshow(C2, sr=sampling_rate, x_axis='time', y_axis='chroma',cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.show()
