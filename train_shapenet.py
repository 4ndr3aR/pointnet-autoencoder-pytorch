#!/usr/bin/env python3

import torch 
import argparse
import os

from torch.utils.data import DataLoader
from torch import optim

#from dataset.dataset import PointCloudDataset
#from dataset.shapenetdataset import ShapeNetDataset
from dataset.SymmetryNet import SymmetryNet

from model.model import PCAutoEncoder
from chamfer_distance.chamfer_distance_cpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance

from dataset.transforms.ComposeTransform import ComposeTransform
from dataset.transforms.RandomSampler import RandomSampler
from dataset.transforms.UnitSphereNormalization import UnitSphereNormalization

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size"  , type=int, default=32    , help="input batch size")
parser.add_argument("--num_points"  , type=int, default=10000 , help="Number of Points to sample")
parser.add_argument("--num_workers" , type=int, default=1     , help="Number Multiprocessing Workers")
parser.add_argument("--dataset_path", type=str, default='/tmp', help="Path to Dataset")
parser.add_argument("--nepoch"      , type=int, default=500   , help="Number of Epochs to train for")

args = parser.parse_args()
print(f"Input Arguments : {args}")

num_points = args.num_points
point_dim  = 3

# Creating Dataset
# train_ds = PointCloudDataset(args.dataset_path, args.num_points, 'train')
# test_ds = PointCloudDataset(args.dataset_path, args.num_points, 'test')

scaler  = UnitSphereNormalization()
sampler = RandomSampler(sample_size=num_points, keep_copy=True)
compose_transform = ComposeTransform([sampler, scaler])


'''
train_ds = ShapeNetDataset(args.dataset_path, args.num_points, split='train')
test_ds  = ShapeNetDataset(args.dataset_path, args.num_points, split='test')
'''
sym_ds   = SymmetryNet(args.dataset_path, compose_transform)
train_ds = sym_ds.train_set
test_ds  = sym_ds.valid_set
sym_ds.train()

def collate_fn(data, debug=False):
	pcd_lst, lbl_lst = zip(*data)
	if debug:
		print(f'{pcd_lst = }')
		print(f'{lbl_lst = }')
	pcdt = torch.stack(pcd_lst)
	lblt = torch.stack(lbl_lst)
	if debug:
		print(f'{pcdt.shape}\n{pcdt = }')
		print(f'{lblt.shape}\n{lblt = }')
	return pcdt, lblt

# Creating DataLoader 
#train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
train_dl = DataLoader(sym_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)


# Output of the dataloader is a tensor reprsenting
# [batch_size, num_channels, height, width]

# getting one data sample
sample = next(iter(train_dl))
print(f'{sample[0].shape = } - {sample[1]}\n{sample[0] = }')

# Creating Model

autoencoder = PCAutoEncoder(point_dim, num_points)

# It is recommented to move the model to GPU before constructing optimizers for it. 
# This link discusses this point in detail - https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/8
# Moving the Network model to GPU
# autoencoder.cuda()

# Setting up Optimizer - https://pytorch.org/docs/stable/optim.html 
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# create folder for trained models to be saved
os.makedirs('saved_models', exist_ok=True)

# create instance of Chamfer Distance Loss Instance
chamfer_dist = ChamferDistance()

# Start the Training 
for epoch in range(args.nepoch):
    for i, data in enumerate(train_dl):
        points, labels = data
        points = points.transpose(2, 1)
        # points = points.cuda()
        optimizer.zero_grad()
        reconstructed_points, global_feat = autoencoder(points)

        dist1, dist2 = chamfer_dist(points, reconstructed_points)
        train_loss = (torch.mean(dist1)) + (torch.mean(dist2))

        print(f"Epoch: {epoch}, Iteration#: {i}, Train Loss: {train_loss}")
        # Calculate the gradients using Back Propogation
        train_loss.backward() 

        # Update the weights and biases 
        optimizer.step()

    scheduler.step()
    torch.save(autoencoder.state_dict(), 'saved_models/autoencoder_%d.pth' % (epoch))
