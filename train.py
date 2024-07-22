import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

import os
import numpy as np
from data_loader.loader import get_data
from models.detector import AD
from models.gcn import Model

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'weight_mixer_minmax.pth'))
        self.val_loss_min = val_loss

device = 'cuda'
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


trajectory_path = "/home/gsong/skeleton_gan_pytorch/data/My-data/training/trajectory/00"
swin = 10
video_shape = (1920,1080,3)
train_ratio = 0.8
batch_size = 2560

Xtrain_local = get_data(trajectory_path,swin)
print(Xtrain_local.shape)
scaler = MinMaxScaler()
scaler.fit(Xtrain_local)
Xtrain_local = scaler.transform(Xtrain_local)
Xtrain = torch.from_numpy(Xtrain_local)

numdata = Xtrain.shape[0]
np.random.seed(0)
randperm = np.random.permutation(numdata)
Xtrain = Xtrain[randperm,:]

train_num = int(np.floor(numdata*train_ratio))
val_num = numdata - train_num
train_dataset = Xtrain[:train_num,:]
val_dataset = Xtrain[train_num:,:]
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                         shuffle=False, num_workers=0)

max_epochs = 1000
criterion = nn.MSELoss()
graph_args = {'strategy':'spatial'}
model = Model(2,34,graph_args).to(device)
optimizer = optim.AdamW(model.parameters(),lr=1e-3)
early_stopping = EarlyStopping(patience=3, verbose=True)

print("==============================Start to train===============================================")

for epoch in range(max_epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        print(data.shape)
        data[:,0,:,:] = data[:,0,:,:]/1920
        data[:,1,:,:] = data[:,1,:,:]/1080
        #xmin, xmax = torch.min(data[:,0,:,:]), torch.max()
        last_pts = torch.cat((data[:,0,-1,:],data[:,1,-1,:]),1).to(device)
        optimizer.zero_grad()
        out = model(data).to(device)
        loss = criterion(out,last_pts)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    for i, data in enumerate(val_loader):
        data = data.to(device)
        data[:,0,:,:] = data[:,0,:,:]/1920
        data[:,1,:,:] = data[:,1,:,:]/1080
        last_pts = torch.cat((data[:,0,-1,:],data[:,1,-1,:]),1).to(device)
        out = model(data).to(device)
        val_loss += criterion(out,last_pts).item()

    print("Val Loss:", val_loss/len(val_loader))
    early_stopping(val_loss, model,'checkpoint')
    if early_stopping.early_stop:
        print("Early stopping")
        break
