import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset


class TestDataset(Dataset):
  def __init__(self,image_blocks,mat,transform = None,phi=0.25):
    self.image_blocks = image_blocks
    self.transform = transform
    self.phi = phi
    self.mat = mat

  def __len__(self):
    return len(self.image_blocks)

  def __getitem__(self,idx):
    image_block = self.image_blocks[idx]
    label = image_block
    if self.transform is not None:
      image_block = self.transform(image_block)
      label = self.transform(label)
    image_block = image_block.view(33*33)
    label = label.view(33*33)
    image_block = image_block.double()
    label = label.double()
    with torch.no_grad():
      image_block = torch.matmul(self.mat,image_block)

    return image_block,label