"""
model by dhiraj inspried from Charles

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAutoEncoder(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: 

    Output: 
    """
    def __init__(self, point_dim, out_num_points, out_point_dim, reshape_last_layer=True, debug=False):
        super(PCAutoEncoder, self).__init__()

        self.out_point_dim = out_point_dim	# because this may be different if we wanna do training on a downstream task (e.g. classification)
        self.reshape_last_layer = reshape_last_layer
        self.debug = debug

        self.conv1 = nn.Conv1d(in_channels=point_dim, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1)

        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=out_num_points*self.out_point_dim)

        #batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
    
    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        out_num_points = x.shape[2]

        if self.debug:
            print(f'ORIG {x.shape = }')

        #encoder
        x = F.relu(self.bn1(self.conv1(x)))
        if self.debug:
            print(f'1.   {x.shape = }')
        x = F.relu(self.bn1(self.conv2(x)))
        if self.debug:
            print(f'2.   {x.shape = }')
        x = F.relu(self.bn1(self.conv3(x)))
        if self.debug:
            print(f'3.   {x.shape = }')
        x = F.relu(self.bn2(self.conv4(x)))
        if self.debug:
            print(f'4.   {x.shape = }')
        x = F.relu(self.bn3(self.conv5(x)))
        if self.debug:
            print(f'5.   {x.shape = }')

        # do max pooling 
        x = torch.max(x, 2, keepdim=True)[0]
        if self.debug:
            print(f'6.   {x.shape = }')
        x = x.view(-1, 1024)
        if self.debug:
            print(f'7.   {x.shape = }')
        # get the global embedding
        global_feat = x

        #decoder
        x = F.relu(self.bn3(self.fc1(x)))
        if self.debug:
            print(f'8.   {x.shape = }')
        x = F.relu(self.bn3(self.fc2(x)))
        if self.debug:
            print(f'9.   {x.shape = }')
        reconstructed_points = self.fc3(x)
        if self.debug:
            print(f'10.  {reconstructed_points.shape = }')

        #do reshaping
        if self.reshape_last_layer:
            reconstructed_points = reconstructed_points.reshape(batch_size, self.out_point_dim, out_num_points)

        return reconstructed_points, global_feat

