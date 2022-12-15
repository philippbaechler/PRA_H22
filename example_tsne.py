# %%
import numpy as np

from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftfreq, rfft, irfft
from scipy.signal import get_window

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

import simpleaudio as sa


# %%
fs, data = wavfile.read('sampleM2left.wav')
t = np.linspace(0, data.size/fs, data.size)
print(fs)

# %%
play_obj = sa.play_buffer(data, 1, 2, fs)
play_obj.wait_done()


# %%
plt.figure(figsize=(18,4))
plt.plot(t, data)


# %%
# Background subtraction?



# %% Data Exploration
# spectrogram
plt.figure(figsize=(18,4))
f, t, S = signal.spectrogram(data, fs, nperseg=24000)
plt.pcolormesh(t, f[0:400], np.log(S[0:400,:]))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.title("Window Length: 500ms")

# %%
# spectrogram
plt.figure(figsize=(18,4))
f, t, S = signal.spectrogram(data, fs, nperseg=24000)
plt.pcolormesh(t, f, np.log(S))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.title("Window Length: 500ms")

# %%
# spectrogram
plt.figure(figsize=(18,4))
f, t, S = signal.spectrogram(data, fs, nperseg=2400)
plt.pcolormesh(t, f, np.log(S))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.title("Window Length: 50ms")

# %%
# spectrogram
plt.figure(figsize=(18,4))
f, t, S = signal.spectrogram(data, fs, nperseg=215)
plt.pcolormesh(t, f, np.log(S))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.title("Window Length: 5ms")

# %% Cutting Audio
CUTTING_LENGTH_ms = 100
NUMBER_OF_SAMPLES = int(CUTTING_LENGTH_ms/1000 * fs)

subsamples = []

for i in range(0, len(data), NUMBER_OF_SAMPLES):
    cut = data[i:i+NUMBER_OF_SAMPLES]
    if len(cut)==NUMBER_OF_SAMPLES:
        subsamples.append(data[i:i+NUMBER_OF_SAMPLES])

print(f"Number of sub samples: {len(subsamples)}")

# %%
play_obj = sa.play_buffer(subsamples[227], 1, 2, fs)
play_obj.wait_done()

# %%
plt.figure(figsize=(18,4))
plt.plot(subsamples[16])

# %%
plt.figure(figsize=(18,4))
plt.plot(subsamples[20])

# %%
plt.figure(figsize=(18,4))
plt.plot(subsamples[41])


# %%
for idx, subsample in enumerate(subsamples):
    subsamples[idx] = (subsample * get_window('hamming', NUMBER_OF_SAMPLES)).astype(int)

# %%
plt.figure(figsize=(18,4))
plt.plot(subsamples[42])



# FFT
# %%
subs = subsamples[20]
S = rfft(subs)
f = fftfreq(subs.size, 1/fs)[:subs.size//2]
plt.figure(figsize=(18,4))
plt.plot(f[:], np.abs(S[:-1].real))
plt.yscale("log")
plt.title("Subsample 25")


# %%
subs = subsamples[16]
S = rfft(subs)
f = fftfreq(subs.size, 1/fs)[:subs.size//2]
plt.figure(figsize=(18,4))
plt.plot(f[:], np.abs(S[:-1].real))
plt.yscale("log")
plt.title("Subsample 25")

# %%
subs = subsamples[41]
S = rfft(subs)
f = fftfreq(subs.size, 1/fs)[:subs.size//2]
plt.figure(figsize=(18,4))
plt.plot(f[:], np.abs(S[:-1].real))
plt.yscale("log")
plt.title("Subsample 41")


# %%
subsamples_f = []

for idx, subsample in enumerate(subsamples):
    subsamples_f.append(np.abs(rfft(subsample).real))

np.array(subsamples_f).shape


# %%
plt.hist((subsamples_f[0]), bins=50)
plt.yscale("log")

# %%
subsamples_f = MinMaxScaler().fit_transform(subsamples_f)



# %%
tsne = TSNE(
            n_components=2,
            perplexity=10,
            n_iter=10000,
            n_iter_without_progress=300,
            n_jobs=2,
            random_state=0
        )

projection = tsne.fit_transform(subsamples_f)        


# %%
plt.figure(figsize=(12,10))
plt.scatter(projection[:, 0], projection[:, 1], c="k", alpha=0.5)

# water pump / extraction
plt.scatter(np.concatenate((projection[54:86, 0], projection[259:494, 0])), \
    np.concatenate((projection[54:86, 1], projection[259:494, 1])), c="g", alpha=0.5)

# grinding
plt.scatter(projection[176:225, 0], projection[176:225, 1], c="gold", alpha=0.5)

# outlet motor down
plt.scatter(projection[16:35, 0], projection[16:35, 1], c="pink", alpha=0.5)

# outlet motor up
plt.scatter(projection[527:545, 0], projection[527:545, 1], c="cyan", alpha=0.5)

# brew chamber motor
plt.scatter(np.concatenate((projection[124:135, 0], projection[139:143, 0], projection[146:149, 0], \
                            projection[230:240, 0], projection[546:560, 0], projection[562:567, 0], \
                            projection[568:572, 0])), \
            np.concatenate((projection[124:135, 1], projection[139:143, 1], projection[146:149, 1], \
                            projection[230:240, 1], projection[546:560, 1], projection[562:567, 1], \
                            projection[568:572, 1])), c="r", alpha=0.5)

# valves
plt.scatter(projection[[12, 40, 90, 100, 227, 252, 496], 0], \
            projection[[12, 40, 90, 100, 227, 252, 496], 1], c="m", alpha=0.5)

plt.legend(["No Label", "Water Pump", "Coffee Grinding", "Outlet Motor Down", "Outlet Motor Up", "Brew Chamber Motor", "Valves"], loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("t-SNE on CoffeeMachine Sound", fontsize=20)
plt.xlabel("t-SNE_1")
plt.ylabel("t-SNE_2")
plt.savefig('tsne.pdf', bbox_inches='tight')  
# %%
