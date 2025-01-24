import os
from paths import (root_path, ust_data_dir, dataset_dir, annotation_file, 
                   taxonomy_file, log_mel_spec_dir)

import librosa
import os
import numpy as np
import pandas as pd
import mel_features
import vggish_params
from IPython.display import clear_output

if not(os.path.isdir(log_mel_spec_dir)):
    os.makedirs(log_mel_spec_dir)

# Some parameters
sr = vggish_params.SAMPLE_RATE
window_length_secs = vggish_params.STFT_WINDOW_LENGTH_SECONDS
hop_length_secs = vggish_params.STFT_HOP_LENGTH_SECONDS
window_length_samples = int(round(sr * window_length_secs))
hop_length_samples = int(round(sr * hop_length_secs))

num_samples = 10*sr # 10-seconds audio clips
num_frames = 1 + int(np.floor((num_samples - window_length_samples) /
                              hop_length_samples))

# How to save the features?
# We have two options :
# --> 'individual': we create a small .npy file containing the log-mel
# spectrogram (numpy array) for each audio file in the dataset. So we have as
# many .npy files as the number of audio examples in the dataset.
# --> 'global':  we create a huge .npy file containing the log-mel spectrograms
# (numpy array) for all the audio files in the dataset.

how_to_save = 'global' # 'global' or 'individual'

# Create a Pandas DataFrame from the annotation CSV file
annotation_data = pd.read_csv(annotation_file).sort_values('audio_filename')

# Create a new frame which only corresponds to the list of audio files
df_audio_files = annotation_data[['split', 'audio_filename']].drop_duplicates()

# List of all audio files
file_list = annotation_data['audio_filename'].unique().tolist()

# Create dictionnary for making the correspondance between splits and
# directories
split2dir = {'train': 'audio-dev/train',
             'validate': 'audio-dev/validate',
             'test': 'audio-eval'}

counter = 0
# Iterate over DataFrame rows as (index, row) pairs, where 'index' is the index
# of the row and 'row' contains the data of the row as a pandas Series
log_mel_spec_list = []
for index, row in df_audio_files.iterrows():
    clear_output(wait=True)

    filename = row['audio_filename']

    print('({}/{}) {}'.format(counter+1, len(df_audio_files), filename))

    partition = row['split']

    audio_path = os.path.join(dataset_dir, split2dir[partition], filename)

    x, sr = librosa.load(audio_path, mono=True, sr=None)
    x =x.T
    log_mel_spec = mel_features.waveform_to_log_mel_spectrogram(x, sr)

    if log_mel_spec.shape[0] < num_frames:
        # add zeros so that the final number of frames is 998
        padding_len = num_frames-log_mel_spec.shape[0]
        zero_pad = np.zeros((padding_len, log_mel_spec.shape[1]))
        log_mel_spec = np.vstack((log_mel_spec, zero_pad))

    elif log_mel_spec.shape[0] > num_frames:
        # remove frames so that the final number of frames is 998
        log_mel_spec = log_mel_spec[:num_frames,:]


    if how_to_save == 'individual':
      data_path = os.path.join(log_mel_spec_dir,
                               os.path.splitext(filename)[0] + '.npy')
      np.save(data_path, log_mel_spec)

    elif how_to_save == 'global':
      log_mel_spec_list.append(log_mel_spec)

    counter+=1

if how_to_save == 'global':
  np.save(os.path.join(log_mel_spec_dir, 'data.npy'), log_mel_spec_list)
