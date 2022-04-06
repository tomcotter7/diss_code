# An explainable method for the detection of Diabetic Retinopathy in 2-D Fundus Images

My UoN 3rd year project. The aim of the project is build an application that takes an input image and produces an output image that highlights the possible symptoms of Diabetic Retinopathy. The accompanying paper is in this repository stored under 20160230_Final_Report.pdf.

## Implementation

### Pre-processing

The dataset for this project was obtained from [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data). I had to balance the dataset as I ran into problems with the number of No-DR images greatly outnumbering the other classes.

The preprocessing performed was an Otsu threshold cropping and resizing the images to be 512x512. I also mapped some data augmentation techniques to the defined datasets.

### Training

All the training was performed on Google Colab, and the final model was saved and downloaded for gradCAM++ implementation. I would recommend doing the same if you would like to train the model on new data. I have included the main.ipynb for ease of use.

I have created this repo to make the code cleaner and easier to understand, however, as specified all the original training and testing code is within main.ipynb. It is a collection of different techniques to train and test data, so may be hectic. This repo tries to clean it up and show my thoughts in a more organised manner.

### Explainabilty

The explainability is performed by gradCAM++, ignoring Guided Backpropagation as it is sub-optimal. Finally, the "explained" image is displayed to the user for download and further analysis.

## Usage

I have included a Makefile for ease of use. ```make run``` executes the application and ```make train``` trains the model with the dataset locations defined in src/training/training.py
