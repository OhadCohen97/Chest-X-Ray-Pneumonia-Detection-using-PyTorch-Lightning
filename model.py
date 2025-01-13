import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from sklearn.metrics import accuracy_score,classification_report
import numpy as np


def evaluate_metric(pred, ans):
    acc = accuracy_score(ans, np.argmax(pred, 1))
    dic = {0:"NORMAL",1:"Pneumonia"}
    print(classification_report(ans, np.argmax(pred, 1), target_names=list(dic.values()),labels=[0, 1]))
    return {"acc": acc}



class ChestXRayCNN(pl.LightningModule):
    def __init__(self,config):
        super(ChestXRayCNN, self).__init__()
        # class_weights = torch.tensor([0.811, 0.189])
        self.cfg = config
        self.criterion = torch.nn.CrossEntropyLoss()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
                                   
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
                                   
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Dropout(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
                                   
        self.conv5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Dropout(0.2))
        # self.conv6 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        #                            nn.ReLU(),
        #                            nn.BatchNorm2d(512),
        #                            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(nn.Flatten(),                       # Flatten the output for FC layer
                                nn.Linear(256,256),               #7x7x256 = 12544
                                nn.ReLU(),
                                nn.Linear(256, 32),
                                nn.Dropout(0.1),
                                nn.ReLU(),
                                nn.Linear(32,config.n_classes),
                                nn.Softmax(dim=1))   


        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.relu = nn.ReLU()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(64 * 56 * 56, 128)
        # self.fc2 = nn.Linear(128, self.cfg.n_classes) 
        # self.softmax = nn.Softmax(dim=1)
    
    # def forward(self, x):
    #     x = self.pool(self.relu(self.conv1(x)))
    #     x = self.pool(self.relu(self.conv2(x)))
    #     x = torch.flatten(x, start_dim=1)
    #     x = self.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x.squeeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y = batch

        x = x.squeeze(1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch= True, prog_bar=True)
  
        return {"y_hat": y_hat, "y": y} 

    def validation_epoch_end(self, outputs):
        pred = torch.cat([d["y_hat"] for d in outputs], dim=0)
        target = torch.cat([d["y"] for d in outputs], dim=0)

        gather_pred = pred.cpu().numpy()
        gather_target = target.cpu().numpy()
        metric_dict = evaluate_metric(gather_pred, gather_target)
        self.log("acc", metric_dict["acc"], on_epoch = True, prog_bar=True, sync_dist=False)
        print(metric_dict, flush = True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)