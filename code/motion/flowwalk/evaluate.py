import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from PIL import Image
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from Regressor import PWCLite
from easydict import EasyDict
from utils.utils import InputPadder, forward_interpolate

@torch.no_grad()
def validate_synblobs(model, iters=24, mode='validation'):
    """ Perform evaluation on the SynBlobs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.SynBlobs(mode=mode)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        im_all = torch.cat([image1, image2], 1)/255.
        flow_pred = model(im_all)['flows_fw']
        flow_pr = flow_pred[0]
        flow_pr[:,0,:,:] *= 255.
        
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    if mode == 'training':
        print("Flow Walk Training SynBlobs EPE: %f" % epe)
    elif mode == 'validation':
        print("Flow Walk Validation SynBlobs EPE: %f" % epe)
    elif mode == 'testing':
        print("Flow Walk Testing SynBlobs EPE: %f" % epe)
    return {'synblobs': epe}

def main(args):
    cfg = EasyDict({"n_frames": 2, "reduce_dense": True, "type": "pwclite", "upsample": True})
    model = torch.nn.DataParallel(PWCLite(cfg))
    model.load_state_dict(torch.load(args.model))
    
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        validate_synblobs(model.module, mode=args.mode)

