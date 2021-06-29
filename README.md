# Deep-Learning
# Bee Subspecies Classification with Transfer Learning

## Introduction
The objective of the project is to use Transfer learning for classification on a dataset with annotated images of bees from varuious location of USA, captured over several months, from various bees subspecies. With the help of Transfer learning model such **mobilenet**, **VGG**, **resnet**, **Inception** and so on, we create the base of model. Then the dataset is split into training, validation and test set. The model created classifies the subspecies of the bees over a period of iteration. Then the model is evaluated by estimating the training error and accuracy and also validation error and validation accuracy. With this details of error rate for classification of bees subspecies, we decide to follow deep learning for image classification tasks. If the model shows a high bias, we will try to improve our model so that model learn better the training dataset. If there is a small bias and high variance, it means the model learned the dataset well but failed to generalize it, which is known as overfitting. Based on such observation , decision for how the model should be adjusted is made. 

## Dataset Content
The dataset was obtained from kaggle website, the dataset contains 5,100+ bee images annotated with location, date, time, subspecies, health condition, caste, and pollen.
From the website it is known that the original batch of images was extracted from still time-lapse videos of bees. By averaging the frames to calculate a background image, each frame of the video was subtracted against that background to bring out the bees in the forefront. The bees are cropped out of the frame so that each image has only one bee. Because each video is accompanied by a form with information about the bees and hive, the labeling process seems to be semi-automated. Each video results has a differing image crop quality levels. 

## Preparation of the data
It is important that we understand the data we have before the processing the dataset. The image can be read, write using libraries such as Skimage, Imageio and matplotlib, plotly for plotting the images and graphs. The dataset we have for classification purpose is a collective data of all the species of bees in USA. The dataset comes with annotation in a seperate csv file. So first we use the library pandas to read and load the dataframe. There are total 5172 images which can be split into 80% for training the model and 20% for testing the model. Further the 80% training dataset is split into 80% for training and 20% for validation purpose. This spliting of dataset can be done using the sklearn library. We will use random_state to ensure the reproducitibility of results.

## Building a Model
The objective of project is to use Transfer learning for object classification. Transfer learning is a process of deploying a pretrained model for the classification problem on a different dataset as first layer with feature extraction capabilities of a converge network. This pretrained model are usually trained on large dataset such as **ImgaeNet** and **COCO**. The intuition of transfer learning is that if the model trained on large and general dataset, it can effectively serve as generic model of a classification task. 


## Feature Extraction 
There are two approaches of transfer learning , Fine Tuning or Retraining and Feature extraction. The approach used for this task is feature extraction, which is simply adding a new output layer, which will be trained from scratch, on top of the pretrained model so that we can repurpose the feature maps learned previously for our dataset and our new output space.

## VGG19 Model
The layer of the VGG19 is used for this task. We repurpose the pretrained model based on the requirement of the task. Here the task output is classifying the bees based on subspecies, which is around 7 types. VGG is a deep CNN with 19 layers. This pretrained model uses ImageNet as its dataset. It is consider as good classification architecture which suits this task conveniently.

## Model Evaluation 
Evaluation of model is a very important part of any deep learning task. Aim is to estimate the generalization accuracy of the model on future data. The evaluation metric used for the task is **F-measure**. F-measure (also F-score) is a measure of a test’s accuracy that considers both the precision and the recall of the test to compute the score. Precision is the number of correct positive results divided by the total predicted positive observations. Recall, on the other hand, is the number of correct positive results divided by the number of all relevant samples (total actual positives).

## Conclusion 
The score obtained from the test dataset is satisfactory as the VGG19 model was good enough for this classifcation task. For further improvment, this problem can be done with other transfer learning model and their accuracy can be compared. This way , the predictions can be made better.

## Software 
For this project the libraries used are **scipy**,**seaborn**, **scikit**,**plotly**,**numpy**,**matplotlib** and **keras**.
The **tensorflow** version used for this task is **1.14** version.
**Python** used for this task is **3.7.10** version.


## Source 
[1] Dataset:  https://www.kaggle.com/jenny18/honey-bee-annotated-images

## References
[1] DeepLearningLecture_Schutera https://github.com/schutera/DeepLearningLecture_Schutera

[2] DanB, CollinMoris, Deep Learning From Scratch, https://www.kaggle.com/dansbecker/deep-learning-from-scratch

[3] Gabriel Preda Bee Classification using CNN  https://www.kaggle.com/gpreda/honey-bee-subspecies-classification
