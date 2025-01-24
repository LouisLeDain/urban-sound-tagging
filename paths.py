import os

root_path = "urban-sound-tagging-project" # YOU SHOULD PUT HERE THE PATH TO THE FOLDER THAT CONTAINS THE README

ust_data_dir = os.path.join(root_path, "data/ust-data")

dataset_dir = os.path.join(ust_data_dir, 'sonyc-ust')

annotation_file = os.path.join(dataset_dir, 'annotations.csv')
taxonomy_file = os.path.join(dataset_dir, 'dcase-ust-taxonomy.yaml')

log_mel_spec_dir = os.path.join(ust_data_dir, 'log-mel-spectrograms')