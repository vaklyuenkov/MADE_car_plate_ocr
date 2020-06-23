# Car plate ocr

Solution based on baseline by Mikhail Gurevich: https://www.kaggle.com/mgurevich/baseline-with-crnn.
Models for detection of car plates and text recognition were trained separately. 
Detection model generate a bounding box for each plate, then rectangle is cut out for recognition.

#### 1. Bbox detection 

Faster R-CNN model with a ResNet-50-FPN backbone
* num_epochs to best model on validation  = 2 
* optimizer - SGD
  * lr=0.005
  * momentum=0.9
  * weight_decay=0.0005
* scheduler - StepLR
  * step_size=3
  * gamma=0.1

#### 2. Plate recognition

CNN for feature extractor and rnn based on GRU for sequence prediction.
CRNN model with ResNet18 backbone.
* num_epochs to best model on validation  = 18
* rnn_num_layers=2
* rnn_hidden_size=128
* dropout=0.3
* loss - ctc_loss
* optimizer- Adam
    lr=3e-4
    weight_decay=1e-4

### Bad ideas:
For recognition cut rectangles with sccore above treshold or rectangle with highest score.

