import torch

from resnet_model import Model
import torch.nn.functional as F
import torch.optim as opt
from config import config
from dataset import train_dataloader,valid_dataloader,test_dataloader
from torch import nn
from tqdm import tqdm

config = config()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

resnet = Model().resnet18
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, 7)
resnet.to(device)



#优化器
optim = opt.Adam(Model().resnet18.parameters(),lr=config.lr)
certifi = F.cross_entropy
schedule = opt.lr_scheduler.StepLR(optimizer=optim,step_size=30,gamma=0.1)#每训练 30 个 epoch 后,学习率乘以0.1

#print(Model.set_requires_grad(resnet))
def train(epochs,optim,certifi,resnet,train_dataloader,valid_dataloader,schedule):
    best_acc = 0
    #print(len(train_dataloader),len(train_dataloader.dataset))  1795 28709
    for epoch in range(epochs):
        loss_ = 0
        loss__ = 0
        train_accurary = 0
        valid_accurary = 0
        total_batch = 0
        for input_,label_ in tqdm(train_dataloader):
            total_batch += 1
            resnet.train()
            optim.zero_grad()
            out = resnet(input_).cpu()
            train_loss = certifi(out,label_)
            train_loss.backward()
            optim.step()
            loss_ += train_loss.item()*len(label_)
            _,out_ = torch.max(out,1)#max,index
            train_accurary += (out_==label_).sum().item()#tensor格式，加减得转格式

        resnet.eval()
        with torch.no_grad():
            for xb,yb in valid_dataloader:
                        out = resnet(xb).cpu()
                        true = yb.data.cpu()
                        valid_loss = certifi(out,true)
                        loss__ += valid_loss.item()*len(yb)
                        evaluate = torch.max(out,1)[1]
                        valid_accurary += (true==evaluate).sum().item()
        schedule.step()#在epoch结束后调用
        if valid_accurary/len(valid_dataloader.dataset) > best_acc:
                best_acc = valid_accurary/len(valid_dataloader.dataset)
                torch.save(resnet.state_dict(),config.save_model)
        print(f'epoch {epoch+1}/{epochs} loss_train {loss_/len(train_dataloader.dataset)} train_acc {train_accurary/len(train_dataloader.dataset)}'
              f'loss_valid {loss__/len(valid_dataloader.dataset)} valid_acc {valid_accurary/len(valid_dataloader.dataset)}'
              )

if __name__ == "__main__":
    train(20,optim,certifi,resnet,train_dataloader,valid_dataloader,schedule)
