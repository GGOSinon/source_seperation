import librosa
import numpy as np
import math
import cmath

def wav2array(filename):
	y, sr = librosa.load(filename)
	return y, sr

def normalize(x):
	pi = math.pi
	if x > pi: return x - 2 * math.pi
	else: return x

def getDeltaPhase(wavfile, n_fft, n_hop, getDelta = True):
	stft_data = librosa.stft(wavfile, n_fft = n_fft, hop_length = n_hop)
	k, m = stft_data.shape[0], stft_data.shape[1]
	phase, A = np.zeros((k, m)), np.zeros((k, m))
	for i in range(k):
		for j in range(m):
			phase[i][j] = cmath.phase(stft_data[i][j])
			A[i][j] = cmath.phase(stft_data[i][j])
	if getDelta == False:
		return [A, phase]
	dtPhase = np.zeros((k, m))
	dfPhase = np.zeros((k, m))
	dtPhaseCorrection = 2 * math.pi * n_hop / n_fft
	for i in range(k-1):
		for j in range(m-1):
			dtPhase[i][j] = phase[i][j+1] - phase[i][j]
			dfPhase[i][j] = phase[i+1][j] - phase[i][j]
	for i in range(k):
		for j in range(m):
			dtPhase[i][j] += dtPhaseCorrection * i
			dfPhase[i][j] -= math.pi
			dtPhase[i][j] = normalize(dtPhase[i][j])
			dfPhase[i][j] = normalize(dfPhase[i][j])
	return [A, phase, dtPhase, dfPhase]

import os
import random

file_list = [[],[],[],[],[]]
for root, dirs, files in os.walk("../DSD100/"):
	for file in files:
		if file.endswith("bass.wav"):
			file_list[0].append(os.path.join(root, file))
		if file.endswith("drums.wav"):
			file_list[1].append(os.path.join(root, file))
		if file.endswith("other.wav"):
			file_list[2].append(os.path.join(root, file))
		if file.endswith("vocals.wav"):
			file_list[3].append(os.path.join(root, file))
		if file.endswith("mixture.wav"):
			file_list[4].append(os.path.join(root, file))

file_num = 100
n_fft, n_hop = 4096, 1024
data_size = 10000
num_channel = 5
num_dataset = 100
dataset = []

for i in range(file_num):
	wavfiles = []
	for channel in range(num_channel):
		y, sr = librosa.load(file_list[channel][i])
		len_music = len(y)
		wavfiles.append(y)
	for k in range(num_dataset):
		if k % 10 == 0: print(k)
		pos = random.randrange(0, len_music - data_size)
		data = []
		for i in range(num_channel):
			wavfile = wavfiles[i][pos : pos + data_size]
			datafile = getDeltaPhase(wavfile, n_fft, n_hop, getDelta = False)
			data.append(datafile)
		dataset.append(data)
	break

dataset = np.array(dataset)
np.save("data.npy", dataset)
print(dataset.shape)
