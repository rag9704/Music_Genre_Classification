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
#spf = wave.open(r"C:\Users\Rag9704\Pictures\genres\classical\classical1.wav",'r')

#Extract Raw Audio from Wav File
#signal = spf.readframes(-1)
#signal = np.fromstring(signal, 'Int16')
#fs = spf.getframerate()



#Time=np.linspace(0, len(signal)/fs, num=len(signal))

#plt.figure(1)
#plt.title('Classical')
#plt.plot(Time,signal)
#plt.show()
#sampling_rate=22050

#x_hiphop,sample_rate =librosa.load(r"C:\Users\Rag9704\Pictures\genres\hiphop\hiphop1.wav",sr=sampling_rate,mono=True,duration=5.0)
#x_classical,sample_rate =librosa.load(r"C:\Users\Rag9704\Pictures\genres\classical\classical1.wav",sr=sampling_rate,mono=True,duration=5.0)
#sample_rate,x_hiphop=wav.read(r"C:\Users\Rag9704\Pictures\genres\hiphop\hiphop1.wav")
#sample_rate,x_classical=wav.read(r"C:\Users\Rag9704\Pictures\genres\classical\classical1.wav")
x_hiphop,sample_rate =librosa.load(r"C:\Users\Rag9704\Pictures\genres\hiphop\hiphop1.wav")
x_classical,sample_rate =librosa.load(r"C:\Users\Rag9704\Pictures\genres\classical\classical1.wav")

hip_hop  = np.abs(librosa.stft(x_hiphop))
classical = np.abs(librosa.stft(x_classical))
#window_size=2048

fig=plt.figure(figsize=(12,5))
fig.subplots_adjust(hspace=.5)

plt.subplot(2,1,1)
#S = librosa.feature.melspectrogram(x_hiphop, sr=sampling_rate, n_fft=window_size)
#logS = cr.amplitude_to_db(S)
plt.title('Hip-Hop')
print(librosa.amplitude_to_db(x_hiphop,ref=np.max))
lib.specshow(librosa.amplitude_to_db(x_hiphop,ref=np.max), x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')

plt.subplot(2,1,2)
#S2 = librosa.feature.melspectrogram(x_classical, sr=sample_rate, n_fft=window_size)
#logS2 = cr.amplitude_to_db(S2)
plt.title('Classical')
#lib.specshow(logS2, sr=sampling_rate, x_axis='time', y_axis='mel')
lib.specshow(librosa.amplitude_to_db(x_hiphop,ref=np.max), x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')



#plt.pcolormesh(t1, f1, np.abs(Zxx1))
#plt.specgram(x_hiphop,Fs=sample_rate)








#plt.subplot(2,1,1)
#FFT_hiphop=scipy.fft(x_hiphop)
#FFT_hiphop_Mag=numpy.absolute(FFT_hiphop)
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Amplitude")
#plt.title('Hip-Hop')
#plt.plot(FFT_hiphop_Mag)


#FFT_Classical=scipy.fft(x_classical)
#FFT_Classical_Mag=numpy.absolute(FFT_Classical)
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Amplitude")
#plt.title('Classical')
#plt.plot(FFT_Classical_Mag)
#plt.show()

plt.show()










