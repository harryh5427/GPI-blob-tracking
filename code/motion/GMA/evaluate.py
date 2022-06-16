import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from PIL import Image
import numpy as np
import torch
import imageio

from network import RAFTGMA

import datasets
from utils import flow_viz
from utils import frame_utils

from utils.utils import InputPadder, forward_interpolate

@torch.no_grad()
def validate_synblobs(model, iters=6, mode='validation'):
    """ Perform evaluation on the SynBlobs (test) split """
    model.eval()
    epe_list = []
    
    val_dataset = datasets.SynBlobs(mode=mode)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
    
    epe = np.mean(np.concatenate(epe_list))
    if mode == 'training':
        print("GMA Training SynBlobs EPE: %f" % epe)
    elif mode == 'validation':
        print("GMA Validation SynBlobs EPE: %f" % epe)
    elif mode == 'testing':
        print("GMA Testing SynBlobs EPE: %f" % epe)
    return {'synblobs': epe}

def main(args):
    if args.dataset == 'separate':
        separate_inout_sintel_occ()
        sys.exit()
    
    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))
    
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        validate_synblobs(model.module, iters=args.iters, mode=args.mode)
