import torch

import tqdm
from torch.utils import data
from pathlib import Path
import numpy as np
import json
import cv2
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as fnn

class CarPlatesFragmentsDataset(data.Dataset):
    def __init__(self, root, transforms, split='train', train_size=0.9, alphabet='abc'):
        super(CarPlatesFragmentsDataset, self).__init__()
        self.root = Path(root)
        self.alphabet = alphabet
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
            plates_filename = self.root / 'test_boxes.json'
            with open(plates_filename) as f:
                json_data = json.load(f)
            self.load_test_data(json_data)
            return
            
        raise NotImplemented(f'Unknown split: {split}')
        
    def load_data(self, json_data):
        for i, sample in enumerate(json_data):
            if sample['file'] == 'train/25632.bmp':
                continue
            for box in sample['nums']:
                points = np.array(box['box'])
                x_0 = np.min([points[0][0], points[3][0]])
                y_0 = np.min([points[0][1], points[1][1]])
                x_1 = np.max([points[1][0], points[2][0]])
                y_1 = np.max([points[2][1], points[3][1]])
                if x_0 >= x_1 or y_0 >= y_1:
                    continue
                if (y_1 - y_0) * 20 < (x_1 - x_0):
                    continue
                self.image_boxes.append(np.clip([x_0, y_0, x_1, y_1], a_min=0, a_max=None))
                self.image_texts.append(box['text'])
                self.image_names.append(sample['file'])
        self.revise_texts()
                
    def revise_texts(self):
        wrong = 'АОНКСРВХЕТМУ'
        correct = 'AOHKCPBXETMY'
        for i in range(len(self.image_texts)):
            self.image_texts[i] = self.image_texts[i].upper()
            for (a, b) in zip(wrong, correct):
                self.image_texts[i] = self.image_texts[i].replace(a, b)
            
                
    def load_test_data(self, json_data):
        for i, sample in enumerate(json_data):
            for box in sample['boxes']:
                if box[0] >= box[2] or box[1] >= box[3]:
                    continue
                points = np.array(box)
                self.image_boxes.append(np.clip(points, a_min=0, a_max=None))
                self.image_names.append(sample['file'])
        self.image_texts = None
    
    def __getitem__(self, idx):
        file_name = self.root / self.image_names[idx]
        image = cv2.imread(str(file_name))
        if image is None:
            file_name = self.image_names[idx]
            image = cv2.imread(str(file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        text = ''
        
        if self.image_boxes is not None:
            box = self.image_boxes[idx]
            image = image.copy()[box[1]:box[3], box[0]:box[2]]
            
        if self.image_texts is not None:
            text = self.image_texts[idx]
            
        seq = self.text_to_seq(text)
        seq_len = len(seq)

        output = dict(image=image, seq=seq, seq_len=seq_len, text=text, file_name=file_name)
        
        if self.transforms is not None:
            output = self.transforms(output)
        
        return output
    
    def text_to_seq(self, text):
        """Encode text to sequence of integers.
        Accepts string of text.
        Returns list of integers where each number is index of corresponding characted in alphabet + 1.
        """
        seq = [self.alphabet.find(c) + 1 for c in text]
        return seq

    def __len__(self):
        return len(self.image_names)


class FeatureExtractor(nn.Module):
    
    def __init__(self, input_size=(64, 320), output_len=20):
        super(FeatureExtractor, self).__init__()
        
        h, w = input_size
        resnet = getattr(models, 'resnet18')(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        self.pool = nn.AvgPool2d(kernel_size=(h // 32, 1))        
        self.proj = nn.Conv2d(w // 32, output_len, kernel_size=1)
  
        self.num_output_features = self.cnn[-1][-1].bn2.num_features    
    
    def apply_projection(self, x):
        """Use convolution to increase width of a features.
        Accepts tensor of features (shaped B x C x H x W).
        Returns new tensor of features (shaped B x C x H x W').
        """
        # YOUR CODE HERE
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x
   
    def forward(self, x):
        # Apply conv layers
        features = self.cnn(x)
        
        # Pool to make height == 1
        features = self.pool(features)
        
        # Apply projection to increase width
        features = self.apply_projection(features)
        
        return features

class SequencePredictor(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3, bidirectional=False):
        super(SequencePredictor, self).__init__()
        
        self.num_classes = num_classes        
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional)
        
        fc_in = hidden_size if not bidirectional else 2 * hidden_size
        self.fc = nn.Linear(in_features=fc_in,
                            out_features=num_classes)
    
    def _init_hidden_(self, batch_size):
        """Initialize new tensor of zeroes for RNN hidden state.
        Accepts batch size.
        Returns tensor of zeros shaped (num_layers * num_directions, batch, hidden_size).
        """
        # YOUR CODE HERE
        num_directions = 2 if self.rnn.bidirectional else 1
        return torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
        
    def _prepare_features_(self, x):
        """Change dimensions of x to fit RNN expected input.
        Accepts tensor x shaped (B x (C=1) x H x W).
        Returns new tensor shaped (W x B x H).
        """
        # YOUR CODE HERE
        x = x.squeeze(1)
        x = x.permute(2, 0, 1)
        return x
    
    def forward(self, x):
        x = self._prepare_features_(x)
        
        batch_size = x.size(1)
        h_0 = self._init_hidden_(batch_size)
        h_0 = h_0.to(x.device)
        x, h = self.rnn(x, h_0)
        
        x = self.fc(x)
        return x


class CRNN(nn.Module):
    
    def __init__(self, alphabet,
                 cnn_input_size=(64, 320), cnn_output_len=20,
                 rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0.3, rnn_bidirectional=False):
        super(CRNN, self).__init__()
        self.features_extractor = FeatureExtractor(input_size=cnn_input_size, output_len=cnn_output_len)
        self.sequence_predictor = SequencePredictor(input_size=self.features_extractor.num_output_features,
                                                    hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                                    num_classes=len(alphabet)+1, dropout=rnn_dropout,
                                                    bidirectional=rnn_bidirectional)
    
    def forward(self, x):
        features = self.features_extractor(x)
        sequence = self.sequence_predictor(features)
        return sequence


class Resize(object):

    def __init__(self, size=(320, 64)):
        self.size = size

    def __call__(self, item):
        """Accepts item with keys "image", "seq", "seq_len", "text".
        Returns item with image resized to self.size.
        """
        # YOUR CODE HERE
        item['image'] = cv2.resize(item['image'], self.size, interpolation=cv2.INTER_AREA)
        return item


def pred_to_string(pred, abc):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out

def decode(pred, abc):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs

def collate_fn(batch):
    """Function for torch.utils.data.Dataloader for batch collecting.
    Accepts list of dataset __get_item__ return values (dicts).
    Returns dict with same keys but values are either torch.Tensors of batched images, sequences, and so.
    """
    images, seqs, seq_lens, texts, file_names = [], [], [], [], []
    for sample in batch:
        images.append(torch.from_numpy(sample["image"]).permute(2, 0, 1).float())
        seqs.extend(sample["seq"])
        seq_lens.append(sample["seq_len"])
        texts.append(sample["text"])
        file_names.append(sample["file_name"])
    images = torch.stack(images)
    seqs = torch.Tensor(seqs).int()
    seq_lens = torch.Tensor(seq_lens).int()
    
    batch = {"image": images, "seq": seqs, "seq_len": seq_lens, "text": texts, "file_name": file_names}
    return batch


def train_recognition(crnn, 
      device,
      train_dataloader,
      val_dataloader,
      optimizer,
      writer,
      num_epochs = 19):
  
  crnn.train()
  for i, epoch in enumerate(range(num_epochs)):
        epoch_losses = []

        for j, b in enumerate(train_dataloader):
            images = b["image"].to(device)
            seqs_gt = b["seq"]
            seq_lens_gt = b["seq_len"]


            seqs_pred = crnn(images).cpu()
            log_probs = fnn.log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            loss = fnn.ctc_loss(log_probs=log_probs, 
                                targets=seqs_gt,  
                                input_lengths=seq_lens_pred,  
                                target_lengths=seq_lens_gt) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        writer.add_scalar('Loss/train', np.mean(epoch_losses), i)
        #print(i, np.mean(epoch_losses))
        with open(f'crnn_{epoch}_epochs_new.pth', 'wb') as fp:
          torch.save(crnn.state_dict(), fp)
  return crnn


def validate_recognition(crnn,device, val_dataloader):
  val_losses = []
  crnn.eval()
  for i, b in enumerate(val_dataloader):
      images = b["image"].to(device)
      seqs_gt = b["seq"]
      seq_lens_gt = b["seq_len"]

      with torch.no_grad():
          seqs_pred = crnn(images).cpu()
      log_probs = fnn.log_softmax(seqs_pred, dim=2)
      seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

      loss = fnn.ctc_loss(log_probs=log_probs,  # (T, N, C)
                          targets=seqs_gt,  # N, S or sum(target_lengths)
                          input_lengths=seq_lens_pred,  # N
                          target_lengths=seq_lens_gt)  # N

      val_losses.append(loss.item())

  return np.mean(val_losses)
