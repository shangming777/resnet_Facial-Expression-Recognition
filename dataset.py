from torchvision import datasets,transforms,models
from torch.utils.data import Dataset,DataLoader
from config import config
import numpy as np
from config import config
from torch import nn

'''class emo_dataset(Dataset):
    def __init__(self,data_path,transform=None):
        super.__init__(data_path,transform=None)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len()

    def __getitem__(self, item):'''
transform = {
    'train' : transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),#随机旋转
        transforms.CenterCrop(64),#中心裁剪
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(0.2,0.1,0.1,0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomGrayscale(p=0.025),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    ]
    ),
    'test' : transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(64),#中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ]
    ),
    "valid" : transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(64),#中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ]
    )
}
batch_size = config().batch_size

train_dataset = datasets.ImageFolder(root=config().train_path,transform=transform['train'])
train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

valid_dataset = datasets.ImageFolder(root=config().val_path,transform=transform['valid'])
valid_dataloader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True)

test_dataset = datasets.ImageFolder(root=config().test_path,transform=transform['test'])
test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)



