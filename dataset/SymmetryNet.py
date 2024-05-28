from glob import glob
from torch.utils.data import Dataset
import numpy as np
import os
import torch

import lzma
from pathlib import Path
from typing import Optional, List, Tuple

import time

class SymmetryNet(Dataset):
    def __init__(self, dataroot, transforms=None):
        super(SymmetryNet, self).__init__()
        self.debug       = False
        self.moar_debug  = False
        self.old_dataset = False
        self.dataroot    = Path(dataroot)
        self.transforms  = transforms
        #self.train_set  = list(glob(os.path.join(self.dataroot, 'train', '*.npz')))
        # self.train_set = list(glob(os.path.join(self.dataroot, 'test', '*.npz')))
        #self.test_set   = list(glob(os.path.join(self.dataroot, 'test', '*.npz')))
        self.training    = True

        self.train_set, self.train_len = self.search_partition(self.dataroot, 'train')
        self.valid_set, self.valid_len = self.search_partition(self.dataroot, 'valid')
        self.test_set , self.test_len  = self.search_partition(self.dataroot, 'test')

        self.pcd = self.train_set

        self.LABEL_DICT = {'astroid': 0, 'citrus': 1, 'cylinder': 2, 'egg_keplero': 3, 'geometric_petal': 4, 'lemniscate': 5, 'm_convexities': 6, 'mouth_curve': 7, 'revolution': 8, 'square': 9}
        '''
        self.LABEL_DICT = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6,
                           'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13,
                           'dresser': 14, 'flower': 15, 'glass': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19,
                           'laptop': 20, 'mantel': 21, 'monitor': 22, 'night': 23, 'person': 24, 'piano': 25,
                           'plant': 26, 'radio': 27, 'range': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32,
                           'table': 33, 'tent': 34, 'toilet': 35, 'tv': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}
        '''
        # LABEL = set()
        # for x in self.test_set:
        #     LABEL.add(os.path.split(x)[1].split('_')[0])
        # LABEL = sorted(list(LABEL))
        # LABEL_DICT = {k: v for v, k in enumerate(LABEL)}

        print(f'Load SymmetryNet done, load {len(self.train_set)} items for training, {len(self.valid_set)} items for validation and {len(self.test_set)} items for testing.')

    def search_partition(self, dataroot, partition):
        if self.debug:
            print(f'Searching xz-compressed point cloud files in {self.dataroot}/{partition}...')
        train_set = list(dataroot.rglob(f'{partition}/*/*.xz'))
        train_len = len(train_set)
        if self.debug:
            print(f'{dataroot.name}/{partition}: found {train_len} files:\n{train_set[:5]}\n{train_set[-5:]}\n')
        return train_set, train_len

    def read_points(self, idx: int) -> torch.Tensor:
        """
        Reads the points with index idx.
        :param idx: Index of points to be read.
                    Not to be confused with the shape ID, this is now just the index in self.flist.
        :return: A tensor of shape N x 3 where N is the amount of points.
        """
        fname, cls = self._filename_from_idx(idx)

        with lzma.open(fname, 'rb') as fhandle:
            points = torch.tensor(np.loadtxt(fhandle)).float()

        if self.debug:
            torch.set_printoptions(linewidth=200)
            torch.set_printoptions(precision=3)
            torch.set_printoptions(sci_mode=False)
            print(f'[{idx}]: {points.shape = }\n{points = }')

        return points, cls

    def _parse_sym_file_old_dataset(self, f, line_amount):
        planar_symmetries = []
        for _ in range(line_amount):
            line = f.readline().split(" ")
            #print(f'{line = }')
            line = [x.replace("\n", "") for x in line]
            plane = [float(x) for x in line[0::]]
            planar_symmetries.append(torch.tensor(plane))
        return planar_symmetries

    def _parse_sym_file(self, filename):
        planar_symmetries = []
        axis_continue_symmetries = []
        axis_discrete_symmetries = []

        print(f'Parsing filename: {Path(filename).name}\n')
        torch.set_printoptions(linewidth=200)
        torch.set_printoptions(precision=3)
        torch.set_printoptions(sci_mode=False)

        with open(filename) as f:
            line_amount = int(f.readline())
            if self.old_dataset:
                planar_symmetries = self._parse_sym_file_old_dataset(f, line_amount)
            else:
                for _ in range(line_amount):
                    line = f.readline().split(" ")
                    #print(f'{line = }')
                    line = [x.replace("\n", "") for x in line]
                    if line[0] == "plane":
                        plane = [float(x) for x in line[1::]]
                        planar_symmetries.append(torch.tensor(plane))
                    elif line[0] == "axis" and line[-1] == "inf":
                        plane = [float(x) for x in line[1:7]]
                        axis_continue_symmetries.append(torch.tensor(plane))
                    else:
                        plane = [float(x) for x in line[1::]]
                        axis_discrete_symmetries.append(torch.tensor(plane))

        planar_symmetries = None if len(planar_symmetries) == 0 else torch.stack(planar_symmetries).float()
        axis_continue_symmetries = None if len(axis_continue_symmetries) == 0 else torch.stack(
            axis_continue_symmetries).float()
        axis_discrete_symmetries = None if len(axis_discrete_symmetries) == 0 else torch.stack(
            axis_discrete_symmetries).float()

        if self.debug:
            print(f'Parsed file: {filename}')
            formatted_print = lambda probably_tensor, text: print(f'\t No {text} found.') if probably_tensor is None \
                else print(f'\tGot {probably_tensor.shape[0]} {text}.')
            formatted_print(planar_symmetries, "Plane symmetries")
            formatted_print(axis_discrete_symmetries, "Discrete axis symmetries")
            formatted_print(axis_continue_symmetries, "Continue axis symmetries")

        if self.moar_debug:
            if planar_symmetries is not None:
                print(f'{planar_symmetries.shape = }')
                print(f'{planar_symmetries = }')
            else:
                print(f'{planar_symmetries = } SHOULD NEVER BE NONE!!!!!!!!!!!!!!!!!!!!!!!!!')
            if axis_discrete_symmetries is not None:
                print(f'{axis_discrete_symmetries.shape = }')
                print(f'{axis_discrete_symmetries = }')
            print(f'{axis_continue_symmetries = }')
            if axis_continue_symmetries is not None:
                #print(f'{axis_continue_symmetries.shape = }')
                print(f'{axis_continue_symmetries = }')
        '''
        print(f'{axis_discrete_symmetries.shape = }')
        print(f'{axis_discrete_symmetries = }')
        print(f'{axis_continue_symmetries.shape = }')
        print(f'{axis_continue_symmetries = }')
        '''

        return planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries

    def _filename_from_idx(self, idx: int) -> Tuple[Path, str]:
        if idx < 0 or idx >= len(self.pcd):
            #raise IndexError(f"Invalid index: {idx}, dataset size is: {len(self.pcd)}")
            print(f'Invalid index: {idx}, dataset size is: {len(self.pcd)} - returning: {self.pcd[0]}')
            idx = 0
        fname = self.pcd[idx]
        cls   = fname.parent.name
        if self.debug:
            print(f'Opening file: {fname.name} with class: {cls}')
        return fname, cls		#, str(fname).replace('.xz', '-sym.txt')

    def read_planes(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Read symmetry planes from file with its first line being the number of symmetry planes
        and the rest being the symmetry planes.
        :param idx: The idx of the syms to reads
        :return: A tensor of planes represented by their normals and points. N x 6 where
        N is the amount of planes and 6 because the first 3 elements
        are the normal and the last 3 are the point.
        """
        xz_fname, _ = self._filename_from_idx(idx)
        sym_fname = xz_fname.parent / xz_fname.name.replace('.xz', '-sym.txt')
        return self._parse_sym_file(sym_fname)

    def __len__(self):
        return len(self.pcd)

    def __getitem__(self, index):
        # index = 0
        #x = self.pcd[index]
        time1 = time.time()
        x, label = self.read_points(index)
        syms = self.read_planes(index)
        time2 = time.time()
        '''
        label = os.path.split(x)[1].split('_')[0]
        label = self.LABEL_DICT[label]
        with np.load(x) as npz:
            pcd, norm = npz['pcd'], npz['norm']
            x = torch.from_numpy(np.concatenate([pcd, norm], axis=1)).float()
        '''
        if self.transforms is not None:
            if self.debug:
                print(f'{self.transforms = }')
                print(f'{x.shape = } - {x.dtype = } - {label = }\n{x = }')
            tfm_x = (0, x, None, None, None)			# idx, points, planar_symmetries, axis_continue_symmetries, axis_discrete_symmetries
            tfm_x = self.transforms(*tfm_x)
            x = tfm_x[1]
        time3 = time.time()
        if self.debug:
            print(f'{x.shape = } - {x.dtype = } - {label = }\n{x = }')
        label = self.LABEL_DICT[label]
        time4 = time.time()
        if self.debug:
            print(f'{(time2 - time1):.3f} seconds')
            print(f'{(time3 - time2):.3f} seconds')
            print(f'{(time4 - time3):.3f} seconds')
        return x, torch.tensor([label], dtype=torch.long), torch.tensor(len(syms))

    def train(self):
        self.training = True
        self.pcd = self.train_set

    def valid(self):
        self.training = False
        self.pcd = self.valid_set

    def eval(self):
        self.training = False
        self.pcd = self.test_set
