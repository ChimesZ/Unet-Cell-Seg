import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model import Unet
from data_loader import * 
import numpy as np
import skimage.io as io

pt_PATH = "/home/zhongzh/Experiment/U-Net_Cell_Detection/unet_model.pt"

device = torch.device("cuda")

test_transforms = transforms.ToTensor()

im_root = "/home/zhongzh/zhongzh_data/Cell_segmentation/test/"

def test():
    model = Unet(1,1,Train=False)
    model.load_state_dict(torch.load(pt_PATH))
    test_dataset = TestDataset(im_root, transform=test_transforms)
    dataloaders = DataLoader(test_dataset,batch_size=1)
    model.eval()
    with torch.no_grad():
        for index,x in enumerate(dataloaders):
            i = 0.95
            y = model(x)  
            img_y = torch.squeeze(y).numpy()
            img_y.shape = (512, 512)
            img = np.where(img_y<i, 0, 1)
            print(img_y)
            io.imsave("/home/zhongzh/Experiment/U-Net_Cell_Detection/test_result/" + str(index) + "_predict_raw.png", img_y)
            io.imsave("/home/zhongzh/Experiment/U-Net_Cell_Detection/test_result/" + str(index) + "_predict_threshold={}.png".format(i), img)
if __name__ == "__main__":
    test()


