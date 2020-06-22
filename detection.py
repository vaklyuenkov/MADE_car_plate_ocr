import torch
import torchvision
from torch.utils import data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
import numpy as np
import pandas as pd
import json
import tqdm
import cv2

class CarPlatesDatasetWithRectangularBoxes(data.Dataset):
    def __init__(self, root, transforms, split='train', train_size=0.9):
        super(CarPlatesDatasetWithRectangularBoxes, self).__init__()
        self.root = Path(root)
        self.train_size = train_size
        
        self.image_names = []
        self.image_ids = []
        self.image_boxes = []
        self.image_texts = []
        self.box_areas = []
        
        self.transforms = transforms
        
        if split in ['train', 'val']:
            plates_filename = self.root / 'train.json'
            with open(plates_filename) as f:
                json_data = json.load(f)
            train_valid_border = int(len(json_data) * train_size) + 1 # граница между train и valid
            data_range = (0, train_valid_border) if split == 'train' \
                else (train_valid_border, len(json_data))
            self.load_data(json_data[data_range[0]:data_range[1]]) # загружаем названия файлов и разметку
            return

        if split == 'test':
            plates_filename = self.root / 'submission.csv'
            self.load_test_data(plates_filename, split, train_size)
            return

        raise NotImplemented(f'Unknown split: {split}')
        
    def load_data(self, json_data):
        for i, sample in enumerate(json_data):
            if sample['file'] == 'train/25632.bmp':
                continue
            self.image_names.append(self.root / sample['file'])
            self.image_ids.append(torch.Tensor([i]))
            boxes = []
            texts = []
            areas = []
            for box in sample['nums']:
                points = np.array(box['box'])
                x_0 = np.min([points[0][0], points[3][0]])
                y_0 = np.min([points[0][1], points[1][1]])
                x_1 = np.max([points[1][0], points[2][0]])
                y_1 = np.max([points[2][1], points[3][1]])
                boxes.append([x_0, y_0, x_1, y_1])
                texts.append(box['text'])
                areas.append(np.abs(x_0 - x_1) * np.abs(y_0 - y_1))
            boxes = torch.FloatTensor(boxes)
            areas = torch.FloatTensor(areas)
            self.image_boxes.append(boxes)
            self.image_texts.append(texts)
            self.box_areas.append(areas)
        
    
    def load_test_data(self, plates_filename, split, train_size):
        df = pd.read_csv(plates_filename, usecols=['file_name'])
        for row in df.iterrows():
            self.image_names.append(self.root / row[1][0])
        self.image_boxes = None
        self.image_texts = None
        self.box_areas = None
         
    
    def __getitem__(self, idx):
        target = {}
        if self.image_boxes is not None:
            boxes = self.image_boxes[idx].clone()
            areas = self.box_areas[idx].clone()
            num_boxes = boxes.shape[0]
            target['boxes'] = boxes
            target['area'] = areas
            target['labels'] = torch.LongTensor([1] * num_boxes)
            target['image_id'] = self.image_ids[idx].clone()
            target['iscrowd'] = torch.Tensor([False] * num_boxes)
#             target['texts'] = self.image_texts[idx]

        image = cv2.imread(str(self.image_names[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)
      
        return image, target

    def __len__(self):
        return len(self.image_names)


def create_model(device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)


def collate_fn1(batch):
    return tuple(zip(*batch))


def train_detection(model,
          device,
          train_loader,
          val_loader,
          num_epochs = 3, 
          writer = None, 
          weight_decay=0.0005, 
          optimizer_step_size=3, 
          gamma=0.1):

  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005,
                              momentum=0.9, weight_decay=weight_decay)
  
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=optimizer_step_size,
                                                gamma=gamma)
  for epoch in range(num_epochs):
      model.train()

      for images, targets in tqdm.tqdm(train_loader):
          images = list(image.to(device) for image in images)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
          loss_dict = model(images, targets)
          losses = sum(loss for loss in loss_dict.values())

          optimizer.zero_grad()
          losses.backward()
          optimizer.step()
      
      batch_losses = []

      for images, targets in tqdm.tqdm(val_loader):
          images = list(image.to(device) for image in images)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
          loss_dict = model(images, targets)
          losses = sum(loss for loss in loss_dict.values())
          batch_losses.append(losses.item())
          optimizer.zero_grad()
      
      batch_losses = np.array(batch_losses)
      batch_losses = batch_losses[np.isfinite(batch_losses)]
      writer.add_scalar('Loss/valid', np.mean(batch_losses), epoch)
      lr_scheduler.step()

      with open(f'fasterrcnn_resnet50_fpn_{epoch}_epoch', 'wb') as fp:
        torch.save(model.state_dict(), fp)

      print(f'validation loss: {np.mean(batch_losses)}')
  return model
  



  