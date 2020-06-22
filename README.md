# Car plate ocr

Detection and recognition

## 1. Bbox detection 

torchvision.models.detection.fasterrcnn_resnet50_fpn

## 2. Plate recognition

A separate model for number recognition. CNN for feature extractor and rnn based on GRU for sequence prediction. 
