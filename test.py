import torch
from config import config
from resnet_model import Model
from torch import nn
from dataset import test_dataloader
import torch.nn.functional as F
import torch.optim as opt


config = config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = Model().resnet18
in_features = resnet.fc.in_features
resnet.fc = nn.Linear(in_features, 7)
resnet.to(device)

certifi = F.cross_entropy


def test(config,resnet,test_dataloader,device,certifi):
    #print(config.save_model)  archive//data//resnet18.ckpt
    state_dict = torch.load(config.save_model)
    resnet.load_state_dict(state_dict)
    resnet.eval()
    test_acc = 0
    test_loss = 0
    with torch.no_grad():
        for input_,label_ in test_dataloader:
            input_ = input_.to(device)
            label_ = label_.to(device)
            out = resnet(input_)
            loss = certifi(out,label_).item()
            test_loss += loss*len(label_)
            true = torch.max(out,1)[1]
            test_acc += (true==label_).sum().item()
    print(f"test_acc {test_acc/len(test_dataloader.dataset)} test_loss {test_loss/len(test_dataloader.dataset)}")


if __name__ == "__main__":
    test(config,resnet,test_dataloader,device,certifi)