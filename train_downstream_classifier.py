#!/usr/bin/env python3

import torch 
import argparse
import os
import sys

from torch.utils.data import DataLoader
from torch import optim

from collections import OrderedDict

#from dataset.dataset import PointCloudDataset
#from dataset.shapenetdataset import ShapeNetDataset
from dataset.SymmetryNet import SymmetryNet

from model.model import PCAutoEncoder
from chamfer_distance.chamfer_distance_cpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance

from dataset.transforms.ComposeTransform import ComposeTransform
from dataset.transforms.RandomSampler import RandomSampler
from dataset.transforms.UnitSphereNormalization import UnitSphereNormalization

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size"  , type=int, default=32         , help="input batch size")
parser.add_argument("--num_points"  , type=int, default=10000      , help="Number of Points to sample")
parser.add_argument("--num_workers" , type=int, default=1          , help="Number Multiprocessing Workers")
parser.add_argument("--dataset_path", type=str, default='/tmp'     , help="Path to Dataset")
parser.add_argument("--nepoch"      , type=int, default=500        , help="Number of Epochs to train for")
parser.add_argument("--nclasses"    , type=int, default=2          , help="Number of classes in the classification problem")
parser.add_argument("--load_model"  , type=str, default='model.pth', help="The pretrained model to be loaded")
parser.add_argument("--freeze"      , type=int, default='0'        , help="Freeze all the layers in the network except the last one")
parser.add_argument("--debug"       , type=int, default='0'        , help="Enable verbose debug prints")

#saved_models/gpo/autoencoder_43.pth

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
	pcd_lst, lbl_lst, nsyms_lst = zip(*data)
	if debug:
		print(f'{pcd_lst   = }')
		print(f'{lbl_lst   = }')
		print(f'{nsyms_lst = }')
	pcdt  = torch.stack(pcd_lst)
	lblt  = torch.stack(lbl_lst)
	nsymt = torch.stack(nsyms_lst)
	if debug:
		print(f'{pcdt.shape}\n{pcdt = }')
		print(f'{lblt.shape}\n{lblt = }')
		print(f'{nsymt.shape}\n{nsymt = }')
	return pcdt, lblt, nsymt

# Creating DataLoader 
#train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
train_dl = DataLoader(sym_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)


# Output of the dataloader is a tensor reprsenting
# [batch_size, num_channels, height, width]

# getting one data sample
print(f'Getting one sample from dataloader...')
sample = next(iter(train_dl))
print(f'{sample[0].shape = } - {sample[1] = } - {sample[2] = }\n{sample[0] = }')

# Creating Model

autoencoder = PCAutoEncoder(point_dim, out_num_points=num_points, out_point_dim=3, reshape_last_layer=False)		# because we have pretrained on 3x10000 input -> 3x10000 output
#classifier  = PCAutoEncoder(point_dim, out_num_points=args.nclasses, out_point_dim=1) # because on the downstream task we want 3x10000 input -> 1x14 output (classification)
print(f'{autoencoder = }')
if args.load_model:
	print(f'Loading model: {args.load_model}')
	#torch.load(args.load_model)
	autoencoder.load_state_dict(torch.load(args.load_model))
	if args.freeze != 0:
		for param in autoencoder.parameters():
				param.requires_grad = False
	fc3_in_features = 1024
	# add more layers as required
	#classifier = torch.nn.Sequential(OrderedDict([('fc3', torch.nn.Linear(fc3_in_features, args.nclasses))]))
	#classifier = torch.nn.Module(OrderedDict([('fc3', torch.nn.Linear(fc3_in_features, args.nclasses))]))
	#classifier = torch.nn.Module(('fc3', torch.nn.Linear(fc3_in_features, args.nclasses)))
	classifier = torch.nn.Linear(fc3_in_features, args.nclasses)
	print(f'Replacing last layer with: {classifier = }')
	autoencoder.fc3 = classifier
	print(f'New pretrained classifier model: {autoencoder = }')

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
crossentropy = torch.nn.CrossEntropyLoss()

# Start the Training 
for epoch in range(args.nepoch):
	for i, data in enumerate(train_dl):
		points, labels, true_nsyms = data
		#print(f'BEFORE {points.shape = }')
		points = points.transpose(2, 1)
		#print(f'AFTER  {points.shape = }')
		# points = points.cuda()
		optimizer.zero_grad()

		pred_nsyms, global_feat = autoencoder(points)
		pred_lbls = torch.argmax(pred_nsyms, dim=1)

		'''
		pred_logits, global_feat = autoencoder(points)
		pred_nsyms = torch.softmax(pred_logits, dim=1)

		#dist1, dist2 = chamfer_dist(points, reconstructed_points)
		#train_loss = (torch.mean(dist1)) + (torch.mean(dist2))

		print(f'{pred_logits.shape = }')
		print(f'{pred_logits = }')
		'''

		if args.debug or i % 100 == 0:
			#print(f'{pred_nsyms.shape = }')
			print(f'{pred_lbls.shape = }')
			print(f'{true_nsyms.shape = }')
			#print(f'{pred_nsyms = }')
			print(f'{pred_lbls = }')
			print(f'{true_nsyms = }')

		train_loss = crossentropy(input=pred_nsyms, target=true_nsyms)

		print(f"Epoch: {epoch}, Iteration#: {i}, Train Loss: {train_loss}")
		# Calculate the gradients using Back Propogation
		train_loss.backward() 

		# Update the weights and biases 
		optimizer.step()

	scheduler.step()
	torch.save(autoencoder.state_dict(), 'saved_models/fine_tuned_autoencoder_%d.pth' % (epoch))
