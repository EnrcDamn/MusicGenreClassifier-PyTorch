# MusicGenreClassifier-PyTorch
A PyTorch re-implementation of the well-known music genre classification exercise.

This algorithm is a deep learning CNN classifier trained on the GTZAN dataset.

## About Dataset

You can download the dataset [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

The GTZAN dataset is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions.

### Content:

* `genres_original` folder - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds.
* `images_original` folder - A visual representation for each audio file. One way to classify data is through neural networks (like I did). Because CNNs usually take in some sort of image representation, the audio files were converted to Mel Spectrograms.
* 2 CSV files - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split into 3 seconds audio files.

Here is a visual representation of the `Data` folder:
```
.
└── Data
    ├── genres_original
    |   ├── blues
    |   ├── classical
    |   └── ...
    ├── images_original
    |   ├── blues
    |   ├── classical
    |   └── ...
    ├── features_3_sec.csv
    └── features_30_sec.csv
```

## Dataset pre-processing
For this exercise I decided not to use the CSV files included in the dataset, and to extract the features I needed myself.
 The `features_30_sec.csv` file was actually modified (to get the filenames and labels) into the `.Data/features_30_sec_final.csv` file, which is the one I used on the script.

I implemented a custom Dataset class, loading it into a DataLoader iterable imported from `torch.utils.data`.
For the audio processing and feature extraction I used Torchaudio, taking advantage of the GPU acceleration. I kept the audio files with their original length (30 seconds), but it could be useful to split the songs into smaller pieces to increase the amount of data we feed into our model.

## Model architecture and training
The model is made of 4 deep convolutional layers; then the output of the convolutional blocks is flattened and feed into a dense layer, and finally passed to the output layer (using Softmax):
```
├─ Sequential (1st conv layer)
├─ Sequential (2nd conv layer)
├─ Sequential (3rd conv layer)
├─ Sequential (4th conv layer)
├─ Flatten
├─ Linear
└─ Softmax
```

Given the exercising nature of the project, the CNN model was trained for 10 epochs.
The pre-trained model is saved as a PTH file and loaded in the inference script.

## Installation

### Prerequisites:
* Python==3.8.x
* torch==1.11.0+cu113
* torchaudio==0.11.0+cu113

Cloning the repository:
```
git clone https://github.com/EnrcDamn/MusicGenreClassifier-PyTorch.git
cd MusicGenreClassifier-PyTorch
```
Then you would need to create a new virtual environment and activate it.

Prerequisites can be installed through the `requirements.txt` file as below:

```
pip install -r requirements.txt
```
