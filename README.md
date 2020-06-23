# Car plate ocr

Solution based on baseline by Mikhail Gurevich: https://www.kaggle.com/mgurevich/baseline-with-crnn
Models for detection of car plates and text recognition were trained separately.

#### 1. Bbox detection 

Faster R-CNN model with a ResNet-50-FPN backbone
* num_epochs to best model on validation  = 2 
* optimizer - SGD
  * lr=0.005
  * momentum=0.9
* scheduler - StepLR
  * weight_decay=0.0005
  * step_size=3
  * gamma=0.1


#### 2. Plate recognition

A separate model for number recognition. CNN for feature extractor and rnn based on GRU for sequence prediction. 
