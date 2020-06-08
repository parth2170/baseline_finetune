import numpy as np
import scipy.io as sio
import torchaudio
import librosa
import os
import sys
import random
from tempfile import NamedTemporaryFile


def load_audio(path):
	sound, _ = librosa.load(path, sr=16000)
	# sound = sound.numpy().T
	# if len(sound.shape) > 1:
	# 	if sound.shape[1] == 1:
	# 		sound = sound.squeeze()
	# 	else:
	# 		sound = sound.mean(axis=1)  # multiple channels, average

	return sound

def augment_audio(path, sample_rate, tempo, gain):
	"""
	Changes tempo and gain of the recording with sox and loads it.
	sudo apt-get update && sudo apt-get install sox libsox-fmt-all
	"""
	with NamedTemporaryFile(suffix=".wav") as augmented_file:
		augmented_filename = augmented_file.name
		sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
		sox_params = "sox -t wav \"{}\" -r {} -c 1 -b 16 -e signed {} {} >/dev/null 2>&1".format(path, sample_rate, augmented_filename, " ".join(sox_augment_params))
		os.system(sox_params)
		y = load_audio(augmented_filename)
		return y

def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15), gain_range=(-6, 8)):
	"""
	Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
	Returns the augmented utterance.
	"""
	low_tempo, high_tempo = tempo_range
	tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
	low_gain, high_gain = gain_range
	gain_value = np.random.uniform(low=low_gain, high=high_gain)
	audio = augment_audio(path=path, sample_rate=sample_rate, tempo=tempo_value, gain=gain_value)
	return audio
