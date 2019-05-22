# Building a Generalized Earthquake Detector: G.E.D 9000

## Idea 

The idea is to feed in Spectrogram images to a convolutional neural network in order to detect earthquakes.

The final goal is to make a Generalized Earthquake Detector, meaning it doesn't need to have seen data in a location to be able to find earthquakes.

## Dataset

Retrieving and processing the data:

https://github.com/jamesaud/seismic-analysis-toolbox


The data is too large to be held on github, but you can download the data with the toolbox. I will make my dataset available in the future (on Dropbox or something).

On my machine I have roughly 4 million waveforms downloaded!


## Code

Written in Pytorch.

If you are familar with Pytorch, the main code will look familiar:

https://github.com/jamesaud/earthquake-detector-9000/blob/master/main.py

The configuration is in json (or dict in Python):

https://github.com/jamesaud/earthquake-detector-9000/blob/master/config.py

In the 'main' code, if the configuration is set to 'environment', `configuration = 'environment'` , then it will read in the configuration from validator/config.json.

## Performance

Use this model for best performance and efficiency:

https://github.com/jamesaud/earthquake-detector-9000/blob/master/models/mnist_3_component.py

It's an interesting model because it feeds 3 components (N, Z, E), meaning three spectrograms, and then combines features to make a detection.

The convnet does 99.0 - 99.9% accuracy on single location QUALITY datasets. 

Many times, the automatic generation of seismic datasets has improperly labeled events or noise. In order to determine what is good quality, use Data Validation. 

## Data Validation (finding good datasets)

To run the conv net on every location, run:

`python data_validate.py`

to produce results that tell you the detection accuracy for each location:

https://github.com/jamesaud/earthquake-detector-9000/blob/master/validator/results_everywhere.csv

## Generalized Model

I kept locations that were 97% accurate and higher from the Data Validation step. I used this as my final dataset to train a generalized earthquake detector.

## Visualization

Run `docker-compose up` to get Tensorboard running on localhost:6000

The code in main is configured to write to the folder for visualization.

## Tests

There are some tests written:

https://github.com/jamesaud/earthquake-detector-9000/tree/master/tests

## Notes

This is a work in progress still. Some of the code (like data_validate.py) does some unconvential things (reloading modules, reading and writing config to a file). 

Part of this is Pytorch seems to be structured for single runs, so when you need to run multiple nets multiple times, it is easier to just reset the module to ensure all weights are set to 0, etc. A big refactor would help my code. Ideally there will be some better wrapper libraries for Pytorch (like Skorch, but better..)

# Installation
Install from requirements.txt (TODO: needs to be updated as an anaconda environment.yml file)
I was running on Cuda75 so I needed install pytorch 0.3.0; in version 0.4.0 there are breaking changes in the code (only took around 10 minutes to update though)

`pip install cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl`