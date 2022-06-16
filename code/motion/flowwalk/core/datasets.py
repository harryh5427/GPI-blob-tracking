# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
    
    def __getitem__(self, index):
        
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        
        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])
        
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)
        
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        
        return img1, img2, flow, valid.float()
    
    
    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class SynBlobs(FlowDataset):
    def __init__(self, aug_params=None, mode='training', root='../data/synthetic_gpi'):
        super(SynBlobs, self).__init__(aug_params)
        
        if mode == 'testing':
            images = sorted(glob(osp.join(root + '/testing', '*.png')))
            flows = sorted(glob(osp.join(root + '/testing', '*.flo')))
            assert (len(images)//2 == len(flows))
            for i in range(len(flows)):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]
        else:
            images = sorted(glob(osp.join(root, '*.png')))
            flows = sorted(glob(osp.join(root, '*.flo')))
            assert (len(images)//2 == len(flows))
            
            split_list = np.loadtxt(root + '/synblobs_split.txt', dtype=np.int32)
            images = images[:2*len(split_list)]
            flows = flows[:len(split_list)]
            for i in range(len(flows)):
                xid = split_list[i]
                if (mode=='training' and xid==1) or (mode=='validation' and xid==2):
                    self.flow_list += [ flows[i] ]
                    self.image_list += [ [images[2*i], images[2*i+1]] ]

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """
    if args.stage == 'synblobs':
        aug_params = None
        train_dataset = SynBlobs(aug_params, mode='training')
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False, shuffle=True, num_workers=4, drop_last=True)
    
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

