from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import random

def image_enhance(img, mask):
    a = random.choice([0,1,2,3])
    if a == 1:
        new_img = transforms.RandomHorizontalFlip(p=1)(img)
        new_mask = transforms.RandomHorizontalFlip(p=1)(mask)
        return new_img, new_mask
    if a == 2:
        new_img = transforms.RandomVerticalFlip(p=1)(img)
        new_mask = transforms.RandomVerticalFlip(p=1)(mask)
        return new_img, new_mask
    if a == 3:
        new_img = transforms.RandomVerticalFlip(p=1)(img)
        new_mask = transforms.RandomVerticalFlip(p=1)(mask)
        new_img = transforms.RandomHorizontalFlip(p=1)(new_img)
        new_mask = transforms.RandomHorizontalFlip(p=1)(new_mask)
        return new_img, new_mask
    if a == 0:
        return img, mask



def train_dataset(img_root, mask_root) -> list:
    '''
    :param img_root: Path of dictionary with raw image.
    :param: mask_root: Path of dictionary with binary (0/255) segmented mask. 
    '''
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n):
        img = os.path.join(img_root, "Data-%d.tif"%(i+1))
        mask = os.path.join(mask_root, "Mask-%d.tif"%(i+1))
        imgs.append((img, mask))
    return imgs

    
class TrainDataset(Dataset):
    def __init__(self, img_root, mask_root, transform = None, target_transform = None, enhance = None):
        super().__init__()
        self.imgs = train_dataset(img_root, mask_root)
        self.transform = transform
        self.target_transform = target_transform
        self.enhance = enhance
    
    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.enhance is not None:
            img, mask = image_enhance(img, mask)
        return img, mask
    def __len__(self):
        return len(self.imgs)


def test_dataset(img_root) -> list:
    imgs = []
    n = len(os.listdir(img_root))
    for i in range(n):
        img = os.path.join(img_root,"test_data_%d.tif" % (i+1))
        imgs.append(img)
    return imgs

class TestDataset(Dataset):
    def __init__(self, img_root, transform = None):
        super().__init__()
        self.imgs = test_dataset(img_root)
        self.transform = transform
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.imgs)


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
 
COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
 
 
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255
