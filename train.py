import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from model import Unet
from data_loader import * 
from tqdm import tqdm
import numpy as np
import time
import json


pt_PATH = "/home/zhongzh/Experiment/U-Net_Cell_Detection/unet_model.pt"
device = torch.device("cuda")

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

mask_tranforms = transforms.ToTensor()

loss_sum = []

def train_model(model, criterion, optimizer, dataload, num_epochs = 10):
    best_model = model
    min_loss = 1000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in tqdm(dataload):
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch %d loss:%0.3f' % (epoch+1, epoch_loss/step))
        if (epoch_loss/step) < min_loss:
            min_loss = (epoch_loss/step)
            best_model = model
            loss_sum.append(epoch_loss/step)
    torch.save(best_model.state_dict(),pt_PATH)
    return best_model

def train():
    model = Unet(1,1).to(device)
    batch_size = 1
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    img_root = "/home/zhongzh/zhongzh_data/Cell_segmentation/Data/"
    mask_root = "/home/zhongzh/zhongzh_data/Cell_segmentation/Mask/"
    enhance = True
    num_epochs = 30

    train_dataset = TrainDataset(img_root,mask_root, transform=img_transforms, target_transform=mask_tranforms,enhance=enhance)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 0)

    train_model(model,criterion,optimizer,dataloader,num_epochs=num_epochs)

    log = {
        'Time': time.asctime( time.localtime(time.time()) ),
        'Model':"Unet",
        'Epoch Number': num_epochs,
        'Batch Size':batch_size,
        'Enhance':enhance,
        'Loss':loss_sum
    }

    with open('/home/zhongzh/Experiment/U-Net_Cell_Detection/log.json','r',encoding='utf-8') as f:
        logs = json.load(f)
        logs.append(log)
        print(json.dumps(logs))
    f.close()
    with open('/home/zhongzh/Experiment/U-Net_Cell_Detection/log.json','w',encoding='utf-8') as f:
        json.dump(logs,f,ensure_ascii=False)
    f.close()


if __name__ == "__main__":
    train()

