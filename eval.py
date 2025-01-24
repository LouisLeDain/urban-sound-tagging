import numpy as np
import os
import shutil
import pandas as pd
import oyaml as yaml
import pytz
import datetime
import json
from tqdm import tqdm
from utils import get_file_targets, get_subset_split, generate_output_file, predict
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc

from paths import (root_path, ust_data_dir, dataset_dir, annotation_file, 
                   taxonomy_file, log_mel_spec_dir)

os.chdir(root_path)

exp_id = 'baseline_2025-01-22-13h35m39s'
output_dir = os.path.join(root_path, 'data/output', exp_id)
results_dir = output_dir

# Create a Pandas DataFrame from the annotation CSV file
annotation_data = pd.read_csv(annotation_file).sort_values('audio_filename')

# List of all audio files
file_list = annotation_data['audio_filename'].unique().tolist()

# Load taxonomy
with open(taxonomy_file, 'r') as f:
    taxonomy = yaml.load(f, Loader=yaml.Loader)

train_file_idxs, val_file_idxs, test_file_idxs = get_subset_split(annotation_data)

# Prediction on test set

print("TEST\n")

prediction_file = os.path.join(output_dir, 'output_predictions_test.csv')

df_dict = evaluate(prediction_file,
                  annotation_file,
                  taxonomy_file,
                  'coarse',
                  'test')

micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)

# Get index of first threshold that is at least 0.5
thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).to_numpy().nonzero()[0][0]

print("Micro AUPRC:           {}".format(micro_auprc))
print("Micro F1-score (@0.5): {}".format(eval_df["F"][thresh_0pt5_idx]))
print("Macro AUPRC:           {}".format(macro_auprc))
print("Coarse Tag AUPRC:")

for coarse_id, auprc in class_auprc.items():
    print("      - {}: {}".format(coarse_id, auprc))

with open(os.path.join(output_dir, 'final_results_test.txt'), 'w') as f:
    f.write("Micro AUPRC:           {}\n".format(micro_auprc))
    f.write("Micro F1-score (@0.5): {}\n".format(eval_df["F"][thresh_0pt5_idx]))
    f.write("Macro AUPRC:           {}\n".format(macro_auprc))
    f.write("Coarse Tag AUPRC:\n")
    for coarse_id, auprc in class_auprc.items():
        f.write("      - {}: {}\n".format(coarse_id, auprc))
