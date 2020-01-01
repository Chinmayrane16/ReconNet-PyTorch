import torch
import os
import cv2
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset

def img_to_blocks(imgs,path,stride=14,filter_size=33):
  images_dataset = []
  for img in imgs:
    image = plt.imread(os.path.join(path,img))
    # 3 dimensions (RGB) convert to YCrCb (take only Y -> luminance)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = image[:,:,0]
    h,w = image.shape
    h_n = ((h - filter_size) // stride) + 1
    w_n = ((w - filter_size) // stride) + 1

    for i in range(h_n):
      for j in range(w_n):
        blocks = image[i*stride:(i*stride)+filter_size, j*stride:(j*stride)+filter_size]
        images_dataset.append(blocks)

  return np.array(images_dataset)


class TrainDataset(Dataset):
  def __init__(self,data_dir,mat,transform = None,phi=0.25):
    self.data_dir = os.listdir(data_dir)
    self.transform = transform
    self.image_blocks = img_to_blocks(self.data_dir,data_dir)
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