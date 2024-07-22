import torch
import torch.nn as nn
import numpy as np
from models.gcn import Model
from data_loader.loader import get_data

import csv
import matplotlib.pyplot as plt

device = 'cuda'
directory = "/home/gsong/skeleton_gan_pytorch/data/My-data2/testing4/trajectory_4/clean"
swin = 10
Xtest = get_data(directory,swin)
Xtest = torch.from_numpy(Xtest).to(device)

Xtest[:,0,:,:] = Xtest[:,0,:,:]/1920
Xtest[:,1,:,:] = Xtest[:,1,:,:]/1080

last_pts = torch.cat((Xtest[:,0,-1,:],Xtest[:,1,-1,:]),1)#.detach().cpu().numpy()
graph_args = {'strategy':'spatial'}
model = Model(2,34,graph_args).to(device)
model.load_state_dict(torch.load("checkpoint/weight.pth"))
total_loss = np.zeros(Xtest.shape[0])

model.eval()

out = model(Xtest)#.detach().cpu().numpy()
total_loss = torch.mean(torch.sqrt((out-last_pts)**2),1,True).cpu().detach().numpy()
with open("/home/gsong/pedestrian/ST-GCN/result/t4.csv",'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for i in range(0,total_loss.shape[0]-1):
                
                
                #timer.elapsed_time()
                csvwriter.writerow(total_loss[i])
                #-------------------------------------------------------------------
            
print('DONE!') 
#plt.plot(loss)
#plt.show()
#print(loss.shape)