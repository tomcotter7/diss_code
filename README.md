# An explainable method for the detection of Diabetic Retinopathy in 2-D Fundus Images

My UoN 3rd year project. The aim of the project is build an application that takes an input image and produces an output image that highlights the possible symptoms of Diabetic Retinopathy. The accompanying paper is in this repository stored under 20160230_dissertation.pdf. The demo video of the working application can be seen in 20160230_demo.mp4.

## Implementation

### Pre-processing

The dataset for this project was obtained from [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection/data). I had to balance the dataset as I ran into problems with the number of No-DR images greatly outnumbering the other classes.

The preprocessing performed was an Otsu threshold cropping and resizing the images to be 512x512. I also mapped some data augmentation techniques to the defined datasets.

### Training

All the training was performed on Google Colab, and the final model was saved and downloaded for gradCAM++ implementation. I would recommend doing the same if you would like to train the model on new data. I have included the explainable_dr_detection.ipynb for ease of use. All that is required is to download the Kaggle dataset from [here](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/overview). The hyper-parameters in explainable_dr_detection.ipynb are optimised to produce a GradCAM++ output. Read the paper for more details on this.

Simply change the directory variables to suit your system and then you can resize and sort the data into the correct structure. Then training can begin using the specified sorted directory.

I have created this repo to make the code cleaner and to integrate the deep learning model into the application, however, as specified all the original training and testing code is within explainable_dr_detection.ipynb. It is easy to read and contains the GradCAM++ implementation so if you just want to use that I recommend looking at that.

### Explainabilty

The explainability is performed by gradCAM++, ignoring Guided Backpropagation as it is sub-optimal. Finally, the "explained" image is displayed to the user for download and further analysis.

### Application

Whilst explainable_dr_detection.ipynb contains all the code needed to train and generate the model, this repo also contains code to build a Tkniter application with the deep learning model built into it

## Usage

I have included a Makefile for ease of use. ```make run``` executes the application and ```make train``` trains the model with the dataset locations defined in src/training/training.py
