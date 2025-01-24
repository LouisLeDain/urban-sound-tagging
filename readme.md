# Urban Sound Tagging Project

<center>
<img width= 500 src='./audio-tagging.png' />
</center>


## Introduction

In this project, you will develop a urban sound tagging system. Given a ten-second audio recording from some urban environment, it will return whether each of the following eight predefined audio sources is audible or not:

1. engine    
2. machinery-impact   
3. non-machinery-impact    
4. powered-saw    
5. alert-signal    
6. music    
7. human-voice    
8. dog

This is a [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) problem.

### Context

The city of New York, like many others, has a "noise code". For reasons of comfort and public health, jackhammers can only operate on weekdays; pet owners are held accountable for their animals' noises; ice cream trucks may play their jingles while in motion, but should remain quiet once they've parked; blasting a car horn is restricted to situations of imminent danger. The noise code presents a plan of legal enforcement and thus mitigation of harmful and disruptive types of sounds.

In an effort towards reducing urban noise pollution, the engagement of citizens is crucial, yet by no means sufficient on its own. Indeed, the rate of complaints that are transmitted, in a given neighborhood, through a municipal service such as 3-1-1, is not necessarily proportional to the level of noise pollution in that neighborhood. In the case of New York City, the Department of Environmental Protection is in charge of attending to the subset of noise complaints which are caused by static sources, including construction and traffic. Unfortunately, statistical evidence demonstrates that, although harmful levels of noise predominantly affect low-income and unemployed New Yorkers, these residents are the least likely to take the initiative of filing a complaint to the city officials. Such a gap between reported exposure and actual exposure raises the challenge of improving fairness, accountability, and transparency in public policies against noise pollution.

Source: [DCASE 2019 Task 5](https://dcase.community/challenge2019/task-urban-sound-tagging)

[![SONYC](https://i.ibb.co/0GSSfrr/https-i-ytimg-com-vi-d-JMt-VLUSEg-maxresdefault.jpg)](https://www.youtube.com/watch?v=d-JMtVLUSEg "SONYC")


### Motivation

Noise pollution is one of the topmost quality of life issues for urban residents in the United States. It has been estimated that 9 out of 10 adults in New York City are exposed to excessive noise levels, i.e. beyond the limit of what the EPA considers to be harmful. When applied to U.S. cities of more than 4 million inhabitants, such estimates extend to over 72 million urban residents.

The objectives of [SONYC](https://wp.nyu.edu/sonyc/) (Sounds of New York City) are to create technological solutions for: (1) the systematic, constant monitoring of noise pollution at city scale; (2) the accurate description of acoustic environments in terms of its composing sources; (3) broadening citizen participation in noise reporting and mitigation; and (4) enabling city agencies to take effective, information-driven action for noise mitigation.

SONYC is an independent research project for mitigating urban noise pollution. One of its aims is to map the spatiotemporal distribution of noise at the scale of a megacity like New York, in real time, and throughout multiple years. To this end, SONYC has designed an acoustic sensor for noise pollution monitoring. This sensor combines a relatively high accuracy in sound acquisition with a relatively low production cost. Between 2015 and 2019, over 50 different sensors have been assembled and deployed in various areas of New York City. Collectively, these sensors have gathered the equivalent of 37 years of audio data.

Every year, the SONYC acoustic sensor network records millions of such audio snippets. This automated procedure of data acquisition, in its own right, gives some insight into the overall rumble of New York City through time and space. However, as of today, each SONYC sensor merely returns an overall sound pressure level (SPL) in its immediate vicinity, without breaking it down into specific components. From a perceptual standpoint, not all sources of outdoor noise are equally unpleasant. For this reason, determining whether a given acoustic scene comes in violation of the noise code requires, more than an SPL estimate in decibels, a list of all active sources in the scene. In other words, in the context of automated noise pollution monitoring, the resort to computational methods for detection and classification of acoustic scenes and events (DCASE) appears as necessary.

Source: [SONYC project](https://wp.nyu.edu/sonyc/) and [DCASE 2019 Task 5](https://dcase.community/challenge2019/task-urban-sound-tagging).

## Project overview

You will work in teams. Each team will compete with others to develop the best-performing urban sound tagging system. At the end of the project, all proposed systems will be ranked based on their performance.

### MyDocker

In order to have access to a GPU, you will work in a remote computing environment using [MyDocker](https://mydocker.centralesupelec.fr/) (see the [documentation](https://centralesupelec.github.io/mydocker)). You will find on the [Edunao](https://centralesupelec.edunao.com/) webpage of the course the link to access the docker image we prepared for this project.

- **Classwork**
  
  For the in-class lab sessions, each group is allowed to start one and one single MyDocker environment. This is because we booked only 1 GPU per group. 
  
  Once this environment is started, just copy the URL to share it with your team mates along with the token value that you will find by entering the command `jupyter server list` in a terminal. 
  
  ⚠️ **Important note**

  - If you share an environment within a team, you will all be connected to the same machine, meaning that you will have the same home directory, you will share the same CPU, GPU and RAM resources, etc. This requires some organization for team work! 
  
    For instance, you can execute `nvidia-smi` in a terminal to monitor the use of the GPU and see if you can run multiple trainings of deep learning models in parallel.

  - By default Jupyter Lab does not support collaborative editing of the files, so be careful not to work simultaneously on the same file. You can create copies, or probably even better you can work on different branches of a same Gitlab/Github repository.

  - The data will remain stored on the environment of the person who started it, so you should download what you want to keep at the end of the lab session. Again, working with a GitLab/GitHub repository can help.

- **Homework**
  
  Be reassured, you will be able to work on the project also at home 😉. You can always ask for an access to the MyDocker environment, at any time, but if it's not during the class you will have to wait for the resources to be available. 





## Setup Instructions

- Log in to the MyDocker environement.
- Open a terminal and
  - execute `pwd` to see your current directory, this is your home directory, which you can always access by executing `cd ~`. 
  - execute `ls` to see the files and folders contained in your current directory, there should be a `workdir` folder.
  - execute ```cd workdir``` to change your current directory.
- Clone this Gitlab repository:
  
  ```git clone https://gitlab-research.centralesupelec.fr/sleglaive/urban-sound-tagging-project.git```
- Refresh the file browser on the left panel. You should now be able to see the cloned repository.
- Edit the variable ```root_path``` in ```baseline/paths.py``` so that it corresponds to the path of the folder `urban-sound-tagging-project` in your environment.
- Run the notebook `setup.ipynb` This notebook will install the Python libraries listed in `requirements.txt` (most of them should already be installed) and it will download the [SONYC-UST dataset](https://zenodo.org/record/2590742#.XIkTPBNKjuM).

### Baseline

To help you start with this project, you have in the `baseline` folder the following Jupyter Notebooks:

* `1-preliminaries.ipynb`: This notebook introduces the [SONYC-UST dataset](https://zenodo.org/record/2590742#.XIkTPBNKjuM). You will also get familiar with:
    * how to manipulate and analyze the dataset;
    * how to read, write, play, and visualize audio files;
    * how to compute a log-Mel spectrogram from a raw audio waveform.

* `2-feature-extraction.ipynb`: In this notebook, you will extract the log-Mel spectrograms for all the audio files in the SONYC-UST dataset. It will take a several minutes.

* `3-baseline-dev.ipynb`: This notebook implements and trains a simple baseline to perform urban sound tagging with PyTorch. 

    Inspired by the original [baseline of the DCASE 2019 Challenge - Task 5](https://github.com/sonyc-project/urban-sound-tagging-baseline) proposed by [Cartwright et al. (2019)](https://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Cartwright_4.pdf), this baseline is a simple multi-label logistic regression model, i.e., a separate binary logistic regression model for each of the 8 classes in the SONYC-UST dataset.
    The model takes [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) embeddings as input, which originally returns a 128-dimensional vector given an audio signal of 0.96 seconds. The SONYC-UST audio data samples being 10-second long, we simply and naively compute VGGish embeddings on short nonoverlapping frames and pool them temporally before feeding the resulting representation to the multi-label logistic regression model. VGGish was trained on [AudioSet](https://github.com/tensorflow/models/tree/master/research/audioset), a dataset of over 2 million human-labeled 10-second YouTube video soundtracks, with labels taken from an ontology of more than 600 audio event classes. This represents more than 5 thousand hours of audio.

    The baseline model is trained to minimize the binary cross-entropy loss, using the Adam optimizer. Early stopping on the validation set is used to mitigate overfitting.

    After training, the performance of the baseline model is evaluated using several metrics described in the [baseline of the DCASE 2019 Challenge - Task 5](https://github.com/sonyc-project/urban-sound-tagging-baseline): micro-averaged area under the precision-recall curve (AUPRC), macro-averaged AUPRC, and micro-averaged F1-score.

    Note that the provided implementation of the baseline training is very inefficient, because it requires a pass of the complete dataset through the frozen VGGish model at each epoch. A much more efficient solution would be to extract and store the VGGish embeddings for the whole SONYC UST dataset and then use these embeddings as input data to the multi-label logistic regression model. However, we chose the above inefficient implementation to make it easier for you to modify and build upon the baseline model. 


## Results

The `data/output/baseline_...` folder contains several output files produced by running the notebook `3-baseline-dev.ipynb`. Check them out.


## Rules

During the development stage of your model, you **must** only use the training and validation set of the SONYC-UST dataset. At the end of the project, you will submit your predictions on the test set, which we will evaluate. The submitted methods will be ranked based on their performance.

## Next steps

Your task is now to develop the best-performing urban sound tagging system on the SONYC-UST dataset. To do so, you should probably exploit/combine multiple sources of information, for instance:

1. the knowledge you acquired by watching the video courses on deep learning;
2. the strengths and weaknesses of the baseline, which you should try to identify;
3. the other methods found by exploring the literature (e.g., papers published along with the DCASE challenge, papers that cite the SONYC-UST dataset, etc.).

For instance, when looking at the structure of the baseline model, you will realize that the temporal pooling of VGGish embeddings obtained for different time frames is very naive, and probably limits the overall performance of the system. Would it be possible to use recurrent neural networks (RNN) to aggregate the information over time? Similarly as the encoder network in [Sequence-to-sequence models for machine translation](https://d2l.ai/chapter_recurrent-modern/seq2seq.html#sec-seq2seq)? But aren't those RNN-based models now outperformed by [attention-based models](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html)? Can we build a temporal pooling operation based on attention mechanisms?

Also, looking at the systems submitted to the [DCASE Challenge](http://dcase.community) for the urban sound tagging task will probably give you ideas. For instance, you can have a look at the [submission of Bongjun Kim](http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Kim_107.pdf), which obtained the 3rd best score at the [DCASE 2019 Challenge, task 5](http://dcase.community/challenge2019/task-urban-sound-tagging) using transfer learning from VGGish.

  
