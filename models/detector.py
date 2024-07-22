import torch
import torch.nn as nn
import numpy as np

from models.gcn import Model
from data_loader.utils import normalize_points_with_size, scale_pose

class AD(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.graph_args = {'strategy':'spatial'}
        self.device = device
        self.model = Model(2,34,self.graph_args,True).to(self.device)

    def forward(self, pts, image_size):
        pts = normalize_points_with_size(pts, image_size[0], image_size[1])
        pts = scale_pose(pts)
        #print(pts.shape)
        #pts = np.concatenate((pts, np.expand_dims((pts[:, 0, :] + pts[:, 1, :]) / 2, 1)), axis=1)
        #print("New:",pts.shape)
        

        pts = torch.tensor(pts, dtype=torch.float32)
        pts = pts.permute(0, 3, 1, 2)

        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        mot = mot.to(self.device)
        pts = pts.to(self.device)
        #print(pts.shape)

        out = self.model((pts, mot))

        return out