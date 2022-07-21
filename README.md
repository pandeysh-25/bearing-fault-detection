# Bearing fault detection using Deep Learning and Machine Learning

Using deep learning and machine learning methods to classify different type of machine bearing faults.

We use MATLAB and the [MPFT Challenge](https://www.mfpt.org/fault-data-sets/) dataset to train a neural network using transfer learning, on scalograms generated from the vibration 
signals, and their amplitudes as found in the dataset. MATLAB's [Deep Learning Toolbox](https://in.mathworks.com/products/deep-learning.html) is used to assist us in training the model, which we
observe has a provisional accuracy of 99.42%. Further work is being done and this repository will be updated.

(Update) Another approach has been added, using Python and the [CWRU (Case Western Reserve University)](https://engineering.case.edu/bearingdatacenter/) dataset to train a vanilla Gaussian naive-Bayes classifier, which had an accuracy of 80%.

(This work is done under the supervision and at the undertaking of Electrono Solutions, between May and July 2022)
