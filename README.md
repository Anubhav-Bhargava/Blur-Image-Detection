# Blur-Image-Detection# Image Blur Detection

This repo contains the code for Image Blur Classifcation. Two methods have been used to solve this problem:-

**Pre-requisites**

> The code is developed for Python 2.7 using Jupyter notebook

> All dependencies are mentioned in requirements.txt

**Laplacian Method**

This method is based on this [tutorial](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/). 
The test accuracy achieved using this method for :-

1.NaturalBlurSet -> ~77%
2.DigitalBlurSet -> ~96%

**Deep Learning based Method**

This method uses Convolution Neural Network for the classification.
The test accuracy achieved using this method for :-

1.NaturalBlurSet -> ~61%
2.DigitalBlurSet -> ~62%

***Training***

The training was done on Google Colaboratory. The training data was saved in pickle files (generated using files pre-processing directory) and uploaded on the Cloud.
The final trained model weights were downloaded and tested againsta the evaluation set.

**Dataset**

It is trained on [CERTH Image Blur Dataset](http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip)


