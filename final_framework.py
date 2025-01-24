### Environment setup

SEED = 0
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import wandb

import os
import shutil
import matplotlib.pyplot as plt

import pandas as pd
import oyaml as yaml
import pytz
import datetime
import json
from tqdm import tqdm

from utils import get_file_targets, get_subset_split, generate_output_file, predict
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc

from torchinfo import summary

from paths import (root_path, ust_data_dir, dataset_dir, annotation_file, 
                   taxonomy_file, log_mel_spec_dir)

os.chdir(root_path)

model_name = 'baseline'

tz_Paris = pytz.timezone('Europe/Paris')
datetime_Paris = datetime.datetime.now(tz_Paris)
timestamp = datetime_Paris.strftime("%Y-%m-%d-%Hh%Mm%Ss")

exp_id = model_name + '_' + timestamp

output_dir = os.path.join(root_path, 'data/output', exp_id)
os.makedirs(output_dir)

### logging in to wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="urban sound tagging",
    settings=wandb.Settings(init_timeout=120),
    # track hyperparameters and run metadata
    config={
    "architecture": "UST",
    "dataset": "UST-data",
    "epochs": 10,
    }
)

### Setting up the GPU 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

### Parameters definition

batch_size = 128
num_epochs = 10
learning_rate_vgg1 = 0.0005
learning_rate_vgg2 = 0.000086
learning_rate_gru = 0.005
patience = 20

# save parameters to disk
params = {'annotation_file': annotation_file,
          'taxonomy_file': taxonomy_file,
          'exp_id': exp_id,
          'log_mel_spec_dir': log_mel_spec_dir,
          'learning_rate_vgg1': learning_rate_vgg1,
          'learning_rate_vgg2': learning_rate_vgg2,
          'learning_rate_gru': learning_rate_gru,
          'batch_size': batch_size,
          'batch_size': batch_size,
          'num_epochs': num_epochs,
          'patience': patience}

kwarg_file = os.path.join(output_dir, "hyper_params.json")
with open(kwarg_file, 'w') as f:
    json.dump(params, f, indent=2)

### Data 

# Create a Pandas DataFrame from the annotation CSV file
annotation_data = pd.read_csv(annotation_file).sort_values('audio_filename')

# List of all audio files
file_list = annotation_data['audio_filename'].unique().tolist()

# Load taxonomy
with open(taxonomy_file, 'r') as f:
    taxonomy = yaml.load(f, Loader=yaml.Loader)

# get list of labels from taxonomy
labels = ["_".join([str(k), v]) for k,v in taxonomy['coarse'].items()]

# list of one-hot encoded labels for all audio files
target_list = get_file_targets(annotation_data, labels)

# list of idices for the training, validation and test subsets
train_file_idxs, val_file_idxs, test_file_idxs = get_subset_split(annotation_data)

# number of classes
n_classes = len(labels)

how_saved = 'global' # 'individual' or 'global'

if how_saved == 'global':
  log_mel_spec_list = list(np.load(os.path.join(log_mel_spec_dir, 'data.npy')))

elif how_saved == 'individual':
  # Create a list of log-Mel spectrograms of size 998 frames × 64 Mel-frequency
  log_mel_spec_list = []
  for idx, filename in enumerate(file_list):
      clear_output(wait=True)

      log_mel_path = os.path.join(log_mel_spec_dir, os.path.splitext(filename)[0] + '.npy')
      log_mel_spec = np.load(log_mel_path)
      log_mel_spec_list.append(log_mel_spec)


### Create training and validation data loaders

class MyDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __getitem__(self, index):
		x = torch.Tensor(self.x[index]).to(device)
		y = torch.Tensor(self.y[index]).to(device)
		return (x, y)

	def __len__(self):
		count = self.x.shape[0]
		return count

train_x = []
train_y = []
for idx in train_file_idxs:
    train_x.append(log_mel_spec_list[idx])
    train_y.append(target_list[idx])

perm_train_idxs = np.random.permutation(len(train_x))

train_x = np.array(train_x)[perm_train_idxs]
train_y = np.array(train_y)[perm_train_idxs]

val_x = []
val_y = []
for idx in val_file_idxs:
    val_x.append(log_mel_spec_list[idx])
    val_y.append(target_list[idx])

perm_val_idxs = np.random.permutation(len(val_x))

val_x = np.array(val_x)[perm_val_idxs]
val_y = np.array(val_y)[perm_val_idxs]

# reshape by adding a channel dimension of size 1
# new shape : (num of examples, channel, frames, frequency bands)
train_x = np.reshape(train_x,(-1,1,train_x.shape[1],train_x.shape[2]))
val_x = np.reshape(val_x,(-1,1,val_x.shape[1],val_x.shape[2]))

train_dataset = MyDataset(train_x, train_y)
val_dataset = MyDataset(val_x, val_y)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

data, target = next(iter(train_loader))

### UST Model Definition

'''
For the moment : VGGish model for feature extraction followed by a RNN for guessing
'''

class VGGish_mod(nn.Module):
  def __init__(self):
    super(VGGish_mod, self).__init__()

    self.layer1_conv1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU())
    self.layer2_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.layer3_conv2 = nn.Sequential(
        nn.Conv2d(64, 128,kernel_size=3, stride=1, padding=1),
        nn.ReLU())
    self.layer4_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.layer5_conv3_1 = nn.Sequential(
        nn.Conv2d(128, 256,kernel_size=3, stride=1,padding=1),
        nn.ReLU())
    self.layer6_conv3_2 = nn.Sequential(
        nn.Conv2d(256, 256,kernel_size=3, stride=1,padding=1),
        nn.ReLU())
    self.layer7_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.layer8_conv4_1 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU())
    self.layer9_conv4_2 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU())

  def forward(self, x, print_shape=False):
      
    # Block 1
    out = self.layer1_conv1(x)
    out = self.layer2_pool1(out)


    # Block 2
    out = self.layer3_conv2(out)
    out = self.layer4_pool2(out)


    # Block 3
    out = self.layer5_conv3_1(out)

    out = self.layer6_conv3_2(out)
    out = self.layer7_pool3(out)

    # Block 4
    out = self.layer8_conv4_1(out)

    out = self.layer9_conv4_2(out)

    return out

# Instantiate the model
vggish = VGGish_mod()

#weights file

vggish_weights_file = os.path.join(root_path, 'data/vggish.pth')
pretrained_state_dict = torch.load(vggish_weights_file)

layer_params = ['layer1_conv1.0.weight', 'layer1_conv1.0.bias', 'layer3_conv2.0.weight', 'layer3_conv2.0.bias', 
                'layer5_conv3_1.0.weight', 'layer5_conv3_1.0.bias', 'layer6_conv3_2.0.weight', 'layer6_conv3_2.0.bias', 
                'layer8_conv4_1.0.weight', 'layer8_conv4_1.0.bias', 'layer9_conv4_2.0.weight', 'layer9_conv4_2.0.bias']

pretrained_state_dict_conv = {key: pretrained_state_dict[key] for key in layer_params if key in pretrained_state_dict}

vggish.load_state_dict(pretrained_state_dict_conv)

# Freeze all parameters
for param in vggish.parameters():
    param.requires_grad = False

for param in vggish.layer9_conv4_2.parameters():
    param.requires_grad = True
  
for param in vggish.layer8_conv4_1.parameters():
    param.requires_grad = True

vggish = vggish.to(device)


# Attention layer

class Attention(nn.Module):
  def __init__(self, hidden_size):
    super(Attention, self).__init__()
    self.attention = nn.Linear(hidden_size, 1, bias=False)

  def forward(self, x):
    # x: (batch_size, seq_len, hidden_size)
    weights = self.attention(x)  # (batch_size, seq_len, 1)
    weights = torch.softmax(weights, dim=1)  # (batch_size, seq_len, 1)
    weighted = x * weights  # (batch_size, seq_len, hidden_size)
    context = torch.sum(weighted, dim=1)  # (batch_size, hidden_size)
    return context

attention = Attention(128)
attention = attention.to(device)

# GRU model

class GRUModel(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, attention):
    super(GRUModel, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.output_size = output_size
    self.attention = attention

    self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, batch_first = True)

    self.fc = nn.Linear(hidden_size, output_size)

    self.sigmoid = nn.Sigmoid()

  def forward(self,x):
    out, hn = self.gru(x)
    out = self.attention(out)
    out = self.fc(out)

    return self.sigmoid(out)

gru = GRUModel(input_size = 512*8, hidden_size = 128, output_size = 8, attention = attention)
gru = gru.to(device)

## Complete model

# Define
class UST(nn.Module):
    def __init__(self, vggish, gru):
        super(UST, self).__init__()

        self.vggish = vggish
        self.gru = gru

    def forward(self, x):

        # extract VGGish embeddings
        x = self.vggish(x)
        #reshaping
        x = x.permute(0, 2, 1, 3)  # (124, 128, 512, 8)
        x = x.reshape(-1, 124, 512 * 8)
        # prediction
        output = self.gru(x)

        return output

# Instantiate
model = UST(vggish, gru)
model = model.to(device)

summary(model, (128,1,998,64))

### Training

train_loss_history = []
val_loss_history = []
min_loss = np.inf
min_epoch = -1
patience_counter = 0

loss_function = nn.BCELoss()
optimizer = torch.optim.Adam([
    {'params': model.vggish.layer9_conv4_2.parameters(), 'lr': learning_rate_vgg2},
    {'params': model.vggish.layer8_conv4_1.parameters(), 'lr': learning_rate_vgg1},
    {'params': model.gru.parameters(), 'lr': learning_rate_gru}
])



for epoch in range(num_epochs):


  # Training
  model.train()
  train_loss = []

  for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                        total=len(train_loader)):
    #print(data.shape)
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    model_output = model(data.cuda()).to(device)
    loss = loss_function(model_output, target)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item()*len(data))


  train_loss_history.append(np.sum(train_loss)/len(train_dataset))
  wandb.log({"training_loss": train_loss_history[-1]})

  # Validation
  model.eval()
  val_loss = []

  with torch.no_grad():
    for batch_idx, (data, target) in tqdm(enumerate(val_loader),
                                        total=len(val_loader)):

      data, target = data.to(device), target.to(device)
      model_output = model(data.cuda()).to(device)
      loss = loss_function(model_output, target)
      val_loss.append(loss.item()*len(data))

    val_loss_history.append(np.sum(val_loss)/len(val_dataset))
    wandb.log({"validation_loss": val_loss_history[-1]})

  # model saving
  if min_loss > val_loss_history[-1]:

    # update best loss
    min_epoch = epoch
    min_loss = val_loss_history[-1]

    # save model
    model_path = os.path.join(output_dir, 'best_model.pth')
    best_state_dict = model.state_dict()
    torch.save(best_state_dict, model_path)

  # early stopping
  if len(val_loss_history) >= 2:

    if val_loss_history[-1] > min_loss:
      patience_counter+=1
    else:
      patience_counter = 0

print(min(val_loss_history))

# Enregistrer les meilleures performances
result = {'Minimum validation loss': min(val_loss_history)}

kwarg_file_res = os.path.join(output_dir, "best_result.json")
with open(kwarg_file_res, 'w') as f:
    json.dump(result, f, indent=2)

model.load_state_dict(best_state_dict)

plt.plot(train_loss_history)
plt.plot(val_loss_history)
plt.legend(('train', 'val'))
plt.savefig(os.path.join(output_dir, "loss.pdf"))
plt.close()

# Prediction on validation set

print("VALIDATION\n")

y_pred = predict(log_mel_spec_list, val_file_idxs, model)


generate_output_file(y_pred, val_file_idxs, output_dir, file_list,
                       'predictions_validation', 'coarse', taxonomy)

prediction_file = os.path.join(output_dir, 'output_predictions_validation.csv')

df_dict = evaluate(prediction_file,
                  annotation_file,
                  taxonomy_file,
                  'coarse',
                   'validate')

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

with open(os.path.join(output_dir, 'final_results_validation.txt'), 'w') as f:
    f.write("Micro AUPRC:           {}\n".format(micro_auprc))
    f.write("Micro F1-score (@0.5): {}\n".format(eval_df["F"][thresh_0pt5_idx]))
    f.write("Macro AUPRC:           {}\n".format(macro_auprc))
    f.write("Coarse Tag AUPRC:\n")
    for coarse_id, auprc in class_auprc.items():
        f.write("      - {}: {}\n".format(coarse_id, auprc))


y_pred = predict(log_mel_spec_list, test_file_idxs, model)


generate_output_file(y_pred, test_file_idxs, output_dir, file_list,
                       'predictions_test', 'coarse', taxonomy)

prediction_file = os.path.join(output_dir, 'output_predictions_test.csv')