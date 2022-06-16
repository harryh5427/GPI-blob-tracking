import sys
import os
import os.path as osp
from glob import glob
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

import bz2
import pickle
import _pickle as cPickle
import numpy as np
import torch
import datasets
from utils import flow_viz
from utils import frame_utils
from raft import RAFT
from skimage import measure
from shapely.geometry import Polygon
from matplotlib.path import Path
from scipy.optimize import linear_sum_assignment

def gen_polygon(points):
    if points[0] != points[-1]:
        if points[0][0] == points[-1][0] or points[0][1] == points[-1][1]:
            points.append(points[0])
        else:
            corner_x = points[0][0] if np.abs(points[0][0] - 0.5) > np.abs(points[-1][0] - 0.5) else points[-1][0]
            corner_y = points[0][1] if np.abs(points[0][1] - 0.5) > np.abs(points[-1][1] - 0.5) else points[-1][1]
            points += [(corner_x, corner_y), points[0]]
    polygon = Polygon(points)
    if polygon.is_valid:
        return polygon
    else:
        return polygon.buffer(0)

def get_poly_mask(points, poly, n_x, n_y):
    mask = np.zeros((n_x, n_y))
    if poly.type == 'MultiPolygon':
        for poly_i in list(poly):
            x, y = poly_i.exterior.coords.xy
            mask[Path([(x[j], y[j]) for j in range(len(x))]).contains_points(points).reshape(n_x, n_y)] = 1.
    else:
        x, y = poly.exterior.coords.xy
        mask[Path([(x[j], y[j]) for j in range(len(x))]).contains_points(points).reshape(n_x, n_y)] = 1.
    return mask.astype(bool)

SMOOTH = 1e-6
def iou_numpy(pred, label, img):
    intersection = np.sum(img[(pred == 1) & (label == 1)])
    union = np.sum(img[(pred == 1) | (label == 1)])
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def main(args):
    dataset_pairs = datasets.SynBlobs(mode=args.mode)
    if args.mode == 'testing':
        files = sorted(glob(osp.join('../data/synthetic_gpi/testing', 'synthetic_gpi_*.pbz2')))
        brt, mask = [], []
        for file in files:
            with bz2.BZ2File(file, 'rb') as f:
                data = cPickle.load(f)
            for t in range(np.shape(data['blob_mask'])[3] - 1):
                mask += [(data['blob_mask'][1:,:,:,t]==1) | (data['blob_mask'][1:,:,:,t+1]==1)]
                
            if len(brt) == 0:
                brt = data['brt_true'][:,:,:-1]
            else:
                brt = np.concatenate([brt, data['brt_true'][:,:,:-1]], axis=2)
        
        print_label = 'Testing'
    else:
        files = sorted(glob(osp.join('../data/synthetic_gpi', 'synthetic_gpi_*.pbz2')))
        brt_all, mask_all = [], []
        for file in files:
            with bz2.BZ2File(file, 'rb') as f:
                data = cPickle.load(f)
            for t in range(np.shape(data['blob_mask'])[3] - 1):
                mask_all += [(data['blob_mask'][1:,:,:,t]==1) | (data['blob_mask'][1:,:,:,t+1]==1)]
            
            if len(brt_all) == 0:
                brt_all = data['brt_true'][:,:,:-1]
            else:
                brt_all = np.concatenate([brt_all, data['brt_true'][:,:,:-1]], axis=2)
        
        split_list = np.loadtxt('../data/synthetic_gpi/synblobs_split.txt', dtype=np.int32)
        
        if args.mode == 'training':
            print_label = 'Training'
            label = 1
        elif args.mode == 'validation':
            print_label = 'Validation'
            label = 2
        
        brt = brt_all[:,:,split_list == label]
        mask = []
        for i in range(len(mask_all)):
            if split_list[i] == label:
                mask.append(mask_all[i])
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.module
    model.cuda()
    model.eval()
    n_x, n_y, _ = np.shape(brt)
    x_frame = np.linspace(0., 1., n_x)
    y_frame = np.linspace(0., 1., n_y)
    y_grid, x_grid = np.meshgrid(y_frame, x_frame)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    iou_all = []
    with torch.no_grad():
        for i in range(len(dataset_pairs)):
            image1, image2, _, _ = dataset_pairs[i]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            _, flow_pr = model(image1, image2, iters=24, test_mode=True)
            flo = flow_pr[0].permute(1,2,0).detach().cpu().numpy()
            flo = flow_viz.flow_to_image(flo)
            cmax_arr = np.sum(255-flo, axis=2)**2
            cmax_arr = 255.*cmax_arr / np.max(cmax_arr)
            contours = measure.find_contours(cmax_arr, 50)
            
            mask_label = []
            for i_label in range(len(mask[i])):
                if np.max(mask[i][i_label]) > 0:
                    mask_label.append(mask[i][i_label])
            
            num_pred = len(contours)
            num_label = len(mask_label)
            cost = np.zeros((num_pred, num_label))
            for i_pred in range(len(contours)):
                polygon_pred = gen_polygon([(contours[i_pred][j,0]/n_x, contours[i_pred][j,1]/n_y) for j in range(np.shape(contours[i_pred])[0])])
                mask_pred = get_poly_mask(points, polygon_pred, n_x, n_y)
                for i_label in range(len(mask_label)):
                    cost[i_pred, i_label] = -iou_numpy(mask_pred, mask_label[i_label], brt[:,:,i])
            
            output_indices, label_indices = linear_sum_assignment(cost)
            iou = np.array([-cost[output_indices[j], label_indices[j]] for j in range(len(output_indices))])
            iou = iou[~np.isnan(iou)]
            if len(iou[iou > 0.1]) > 0:
                iou_all.append(np.mean(iou[iou > 0.1]))
    
    print("RAFT " + print_label + " Mean IoU: " + str(np.mean(iou_all)))

