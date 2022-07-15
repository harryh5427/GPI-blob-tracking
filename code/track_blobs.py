import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import imageio

import pickle
import bz2
import _pickle as cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.path import Path

from skimage import measure
import shapely
from shapely.geometry import Polygon
from shapely.geometry import Point
from scipy.optimize import linear_sum_assignment
import time


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

def get_brt_contour(polygon_obj, n_x, n_y, points, brt, perc_contour, brt_minimum):
    mask = get_poly_mask(points, polygon_obj, n_x, n_y)
    return measure.find_contours(brt, np.max([np.max(brt[mask])*perc_contour, brt_minimum]))

def compute_viou(pred_contours, brt, brt_minimum):
    n_x, n_y = np.shape(brt)
    x_frame = np.linspace(0., 1., n_x)
    y_frame = np.linspace(0., 1., n_y)
    y_grid, x_grid = np.meshgrid(y_frame, x_frame)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    vious = []
    polygons_pred = []
    polygons_fwhm = []
    idx_processed = []
    for idx, current_contour in enumerate(pred_contours):
        polygon_pred = gen_polygon([(current_contour[j,0]/(n_x-1), current_contour[j,1]/(n_y-1)) for j in range(np.shape(current_contour)[0])])
        if polygon_pred.area == 0.:
            polygons_pred.append(None)
            polygons_fwhm.append(None)
            vious.append(0.)
            idx_processed.append(idx)
            continue
        fwhm_contours = get_brt_contour(polygon_pred, n_x, n_y, points, brt, 0.7, brt_minimum)
        found = False
        for fwhm_contour in fwhm_contours:
            polygon_fwhm = gen_polygon([(fwhm_contour[j,0]/(n_x-1), fwhm_contour[j,1]/(n_y-1)) for j in range(np.shape(fwhm_contour)[0])])
            intersection = polygon_pred.intersection(polygon_fwhm)
            if intersection.area > 0.:
                polygons_pred.append(polygon_pred)
                polygons_fwhm.append(polygon_fwhm)
                
                union = polygon_pred.union(polygon_fwhm)
                mask_union = get_poly_mask(points, union, n_x, n_y)
                union_v = np.sum(brt[mask_union])
                
                mask_intersection = get_poly_mask(points, intersection, n_x, n_y)
                intersection_v = np.sum(brt[mask_intersection])
                vious.append(intersection_v/union_v)
                if intersection_v/union_v < 0.2:
                    idx_processed.append(idx)
                
                found = True
                break
        if not found:
            polygons_pred.append(None)
            polygons_fwhm.append(None)
            vious.append(0.)
            idx_processed.append(idx)
    
    idx_sort = np.argsort(vious)[::-1]
    polygons_pred_merged = []
    polygons_fwhm_merged = []
    vious_merged = []
    do_not_pair = {}
    for idx_curr in idx_sort:
        do_not_pair[idx_curr] = []
    for idx_curr in idx_sort:
        if idx_curr in idx_processed or vious[idx_curr] == 0.:
            continue
        idx_processed.append(idx_curr)
        polygon_curr = polygons_pred[idx_curr]
        polygon_curr_final = polygon_curr
        polygon_fwhm_final = polygons_fwhm[idx_curr]
        viou_final = vious[idx_curr]
        for idx_other in range(len(pred_contours)):
            if idx_other == idx_curr or idx_other in idx_processed or idx_other in do_not_pair[idx_curr]:
                continue
            polygon_other = polygons_pred[idx_other]
            if polygon_other == None or len(polygon_other.bounds) == 0:
                idx_processed.append(idx_other)
                continue
            
            polygon_curr_merged = polygon_curr.union(polygon_other)
            fwhm_contours_merged = get_brt_contour(polygon_curr_merged, n_x, n_y, points, brt, 0.7, brt_minimum)
            
            polygon_fwhm_other = polygons_fwhm[idx_other]
            intersection_fwhm = polygon_fwhm_other.intersection(polygon_fwhm_final)
            if intersection_fwhm.area > 0.:
                intersection_fwhm_a = intersection_fwhm.area
                union_fwhm_a = polygon_fwhm_other.union(polygon_fwhm_final).area
                if intersection_fwhm_a/union_fwhm_a > 0.8:
                    
                    for fwhm_contour_merged in fwhm_contours_merged:
                        polygon_fwhm_merged = gen_polygon([(fwhm_contour_merged[j,0]/(n_x-1), fwhm_contour_merged[j,1]/(n_y-1)) for j in range(np.shape(fwhm_contour_merged)[0])])
                        intersection_merged = polygon_curr_merged.intersection(polygon_fwhm_merged)
                        if intersection_merged.area > 0.:
                            union_merged = polygon_curr_merged.union(polygon_fwhm_merged)
                            mask_union_merged = get_poly_mask(points, union_merged, n_x, n_y)
                            union_v_merged = np.sum(brt[mask_union_merged])
                            
                            mask_intersection_merged = get_poly_mask(points, intersection_merged, n_x, n_y)
                            intersection_v_merged = np.sum(brt[mask_intersection_merged])
                            break
                    
                    viou_merged = intersection_v_merged/union_v_merged
                    
                    polygon_curr_final = polygon_curr_merged
                    polygon_fwhm_final = polygon_fwhm_merged
                    viou_final = viou_merged
                    idx_processed.append(idx_other)
                else:
                    do_not_pair[idx_other].append(idx_curr)
            else:
                do_not_pair[idx_other].append(idx_curr)
        
        if viou_final > 0.05:
            polygons_pred_merged.append(polygon_curr_final)
            polygons_fwhm_merged.append(polygon_fwhm_final)
            vious_merged.append(viou_final)
    
    return vious_merged, polygons_pred_merged, polygons_fwhm_merged

def process_image(brt, args, device):
    if args.is_opticalFlow:
        img = np.repeat(brt[np.newaxis, :, :], 3, axis=0)
        img = 1. - torch.from_numpy(img).float()
        img = 255.*img[None]
    else:
        img = np.repeat(brt.T[np.newaxis, :, :], 3, axis=0)
        img = torch.from_numpy(img).float()
    return img.to(device)

def try_find_contours(arr, val, n_x, n_y):
    #vals_adjust = [0, -1, +1, -2, +2, -3, +3, -4, +4, -5, +5]
    vals_adjust = [0, -1, +1, -2, +2, -3, +3]
    for val_adjust in vals_adjust:
        try:
            contours = measure.find_contours(arr, val + val_adjust)
            contours_valid = []
            for contour in contours:
                pi_x, pi_y = contour[0].astype(int)
                pf_x, pf_y = contour[-1].astype(int)
                if all(contour[-1] == contour[0]) or (pi_x == n_x-1 and pf_y in [0, n_y-1]) or (pf_x == n_x-1 and pi_y in [0, n_y-1]) or (pi_x == n_x-1 and pf_x == n_x-1) or (pf_x == n_x-1 and pi_x == n_x-1) or (pi_y == 0 and pf_y == 0) or (pi_y == n_y-1 and pf_y == n_y-1):
                    contours_valid.append(contour)
            
            return contours_valid
        except:
            pass
    return []

def run_pred(args, model, device, predict):
    output = []
    output_tracking = {}
    per_frame_pred_times = []
    per_frame_proc_times = []
    with torch.no_grad():
        with bz2.BZ2File(args.filename, 'rb') as f:
            data = cPickle.load(f)
        
        brt_true = data['brt_true']
        shear_contour_x = data['shear_contour_x']
        shear_contour_y = data['shear_contour_y']
        n_x, n_y, n_t = np.shape(brt_true)
        
        blob_id = 0
        active_tracklets = []
        finished_tracklets = []
        x_frame = np.linspace(0., 1., n_x)
        y_frame = np.linspace(0., 1., n_y)
        y_grid, x_grid = np.meshgrid(y_frame, x_frame)
        points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        
        if args.is_opticalFlow:
            num_iter = n_t - 1
        else:
            num_iter = n_t
        
        for t in range(num_iter):
            print('Working on t = ' + str(t))
            if args.is_opticalFlow:
                image1 = process_image(brt_true[:,:,t], args, device)
                image2 = process_image(brt_true[:,:,t+1], args, device)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                tick = time.time()
                flow_up = predict(model, image1, image2)
                tock = time.time()
                per_frame_pred_times.append(tock - tick)
                img = image1[0].permute(1,2,0).detach().cpu().numpy()
                flo = flow_up[0].permute(1,2,0).detach().cpu().numpy()
                # map flow to rgb image
                flo = flow_viz.flow_to_image(flo)
                img_flo = np.concatenate([img, flo], axis=0)/255.0
                output.append(img_flo[:, :, [2,1,0]])
                
                pred_contours = []
                len_prev = 1000
                cmax_arr = np.sum(255-flo, axis=2)**2
                cmax_arr = 255.*cmax_arr / np.max(cmax_arr)
                #for val in range(30, 256, 2):
                for val in range(30, 256, 7):
                    if np.max(cmax_arr) - val < 7:
                        pred_contours += try_find_contours(cmax_arr, val-10, n_x, n_y)
                        break
                    contours = try_find_contours(cmax_arr, val, n_x, n_y)
                    if len(contours) != len_prev:
                        pred_contours += try_find_contours(cmax_arr, val-10, n_x, n_y)
                    
                    len_prev = len(contours)
            else:
                output.append(brt_true[:,:,t])
                image = process_image(brt_true[:,:,t], args, device)
                tick = time.time()
                result = predict(model, image)
                tock = time.time()
                per_frame_pred_times.append(tock - tick)
                pred_masks = np.transpose(result['masks'][result['labels'] == 2][:,0,:,:].detach().cpu().numpy(), (0,2,1))
                pred_contours = []
                for i in range(np.shape(pred_masks)[0]):
                    pred_contours += measure.find_contours(pred_masks[i,:,:], 0.90)
            
            tick = time.time()
            idx_valid_contours = []
            mask_shear = np.zeros((n_x, n_y))
            for i in range(len(shear_contour_y)):
                mask_shear[:shear_contour_x[i], shear_contour_y[i]] = 1.
            
            for i, current_contour in enumerate(pred_contours):
                polygon_pred = gen_polygon([(current_contour[j,0]/(n_x-1), current_contour[j,1]/(n_y-1)) for j in range(np.shape(current_contour)[0])])
                if polygon_pred.area > 0.:
                    mask = get_poly_mask(points, polygon_pred, n_x, n_y)
                    if mask.any() and np.max(brt_true[:,:,t][mask]) > 0.5:
                        cx = n_x*np.average(x_grid[mask], weights=brt_true[:,:,t][mask])
                        cy = n_y*np.average(y_grid[mask], weights=brt_true[:,:,t][mask])
                        
                        num_pixels = len(x_grid[mask == 1.].flatten())
                        num_pixels_inside = len(x_grid[(mask == 1.) & (mask_shear == 1.)].flatten())
                        
                        if cx > shear_contour_x[np.argmin(np.abs(shear_contour_y - cy))] and num_pixels_inside/num_pixels < 0.3:
                            idx_valid_contours.append(i)
            
            pred_contours = [pred_contours[i] for i in idx_valid_contours]
            
            viou_list, polygon_pred_list, polygon_fwhm_list = compute_viou(pred_contours, brt_true[:,:,t], 0.5)
            cx_list, cy_list = [], []
            amp_valid = [False]*len(polygon_pred_list)
            for i, polygon_pred in enumerate(polygon_pred_list):
                pred_mask = get_poly_mask(points, polygon_pred, n_x, n_y).astype(float)
                if np.max(brt_true[:,:,t][pred_mask == 1.]) > args.amp_threshold:
                    amp_valid[i] = True
                    cx = n_x*np.average(x_grid[pred_mask == 1.], weights=brt_true[:,:,t][pred_mask == 1.])
                    cy = n_y*np.average(y_grid[pred_mask == 1.], weights=brt_true[:,:,t][pred_mask == 1.])
                    cx_list.append(cx)
                    cy_list.append(cy)
            
            viou_list = [viou_list[i] for i in range(len(viou_list)) if amp_valid[i]]
            polygon_pred_list = [polygon_pred_list[i] for i in range(len(polygon_pred_list)) if amp_valid[i]]
            polygon_fwhm_list = [polygon_fwhm_list[i] for i in range(len(polygon_fwhm_list)) if amp_valid[i]]
            
            output_tracking[t] = []
            
            prev_polygons = [tracklet['polygon'] for tracklet in active_tracklets]
            cost = np.zeros((len(prev_polygons), len(polygon_pred_list)))
            for prev_idx in range(len(prev_polygons)):
                for curr_idx in range(len(polygon_pred_list)):
                    cost[prev_idx, curr_idx] = -prev_polygons[prev_idx].intersection(polygon_pred_list[curr_idx]).area
            
            # Bipartite matching
            prev_indices, curr_indices = linear_sum_assignment(cost)
            excl_prev_idx, excl_curr_idx = [], []
            for prev_idx, curr_idx in zip(prev_indices, curr_indices):
                if cost[prev_idx, curr_idx] == 0:
                    excl_prev_idx.append(prev_idx)
                    excl_curr_idx.append(curr_idx)
            
            prev_indices = np.array([prev_indices[i] for i in range(len(prev_indices)) if prev_indices[i] not in excl_prev_idx])
            curr_indices = np.array([curr_indices[i] for i in range(len(curr_indices)) if curr_indices[i] not in excl_curr_idx])
            
            # Add matches to active tracklets
            for prev_idx, curr_idx in zip(prev_indices, curr_indices):
                active_tracklets[prev_idx]['polygon'] = polygon_pred_list[curr_idx]
                active_tracklets[prev_idx]['t_history'].append(t)
                active_tracklets[prev_idx]['viou_history'].append(viou_list[curr_idx])
                blob_id_curr = active_tracklets[prev_idx]['id']
                output_tracking[t].append([blob_id_curr, viou_list[curr_idx], cx_list[curr_idx], cy_list[curr_idx], polygon_pred_list[curr_idx], polygon_fwhm_list[curr_idx]])
            # Finalize lost tracklets
            if t == np.shape(brt_true)[2] - 2:
                lost_indices = set(range(len(active_tracklets)))
            else:
                lost_indices = set(range(len(active_tracklets))) - set(prev_indices)
            for lost_idx in sorted(lost_indices, reverse=True):
                if len(active_tracklets[lost_idx]['t_history']) < args.blob_life_threshold or np.mean(active_tracklets[lost_idx]['viou_history']) < args.viou_threshold:
                    for t_past in active_tracklets[lost_idx]['t_history']:
                        blob_id_lost = active_tracklets[lost_idx]['id']
                        output_tracking[t_past] = [output_tracking[t_past][i] for i in range(len(output_tracking[t_past])) if output_tracking[t_past][i][0] != blob_id_lost]
                finished_tracklets.append(active_tracklets.pop(lost_idx))
            # Activate new tracklets
            new_indices = set(range(len(polygon_pred_list))) - set(curr_indices)
            for new_idx in new_indices:
                blob_id += 1
                active_tracklets.append({'polygon':polygon_pred_list[new_idx], 'id':blob_id, 't_history':[t], 'viou_history':[viou_list[new_idx]]})
                output_tracking[t].append([blob_id, viou_list[new_idx], cx_list[new_idx], cy_list[new_idx], polygon_pred_list[new_idx], polygon_fwhm_list[new_idx]])
            
            tock = time.time()
            per_frame_proc_times.append(tock - tick)
    
    return output, output_tracking, np.mean(per_frame_pred_times), np.mean(per_frame_proc_times)

def plot_frame(t, output, output_tracking, axes, args, hand_labels=None):
    global figs_pred, figs_fwhm, figs_id, figs_hl
    print('Working on video t = ' + str(t))
    for track in figs_pred:
        if isinstance(track, list):
            track[0].remove()
        else:
            for coll in track.collections:
                coll.remove()
    figs_pred = []
    for track in figs_fwhm:
        if isinstance(track, list):
            track[0].remove()
        else:
            for coll in track.collections:
                coll.remove()
    figs_fwhm = []
    for track in figs_id:
        track.remove()
    figs_id = []
    for track in figs_hl:
        track.remove()
    figs_hl = []
    
    if args.is_opticalFlow:
        img_flo = output[t]
        img = 1. - img_flo[:256,:,0]
        flo = img_flo[256:256*2,:,:]
        im1 = axes[1].imshow(np.transpose(flo, (1,0,2)), vmin=0, vmax=1, origin='lower')
        axes[1].set_title('Optical flow, PREDICTED by ' + args.model_label)
    else:
        img = output[t]
    
    im0 = axes[0].imshow(img.T, vmin=0, vmax=1, origin='lower')
    axes[0].set_title('t = ' + str(t))
    
    n_x, n_y = np.shape(img)
    x_frame = np.linspace(0., 1., n_x)
    y_frame = np.linspace(0., 1., n_y)
    y_grid, x_grid = np.meshgrid(y_frame, x_frame)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    im2 = axes[-1].imshow(img.T, vmin=0, vmax=1, origin='lower')
    for tracklet in output_tracking[t]:
        blob_id, viou, cx, cy, polygon_pred, polygon_fwhm = tracklet
        if not args.hand_labels:
            if viou > args.viou_threshold:
                fwhm_mask = get_poly_mask(points, polygon_fwhm, n_x, n_y).astype(float)
                figs_fwhm.append(axes[-1].contour(x_grid*n_x, y_grid*n_y, fwhm_mask, colors='b', linewidths=1))
            figs_id.append(axes[-1].text(cx, cy, 'Blob ' + str(blob_id), color='red', fontsize=10, ha='center', va='center', clip_box=axes[-1].clipbox, clip_on=True))
        
        pred_mask = get_poly_mask(points, polygon_pred, n_x, n_y).astype(float)
        figs_pred.append(axes[-1].contour(x_grid*n_x, y_grid*n_y, pred_mask, colors='r', linewidths=2))
    
    axes[-1].set_title('Blob tracking by ' + args.model_label)
    axes[-1].set_xlim(0., n_x)
    axes[-1].set_ylim(0., n_y)
    
    if args.hand_labels:
        for i, hand_label in enumerate(hand_labels):
            if t in hand_label:
                for coord in hand_label[t]:
                    figs_hl.append(axes[-1].scatter(coord[0], coord[1], s=40, c='C' + str(i)))
    
    plt.tight_layout()

def init():
    global args, axes, hand_labels
    axes[-1].plot([], c='r', label=args.model_label)
    if args.hand_labels:
        for i in range(len(hand_labels)):
            axes[-1].scatter([], [], c='C' + str(i), s=40, label='Labeler ' + str(i+1))
            axes[-1].legend(loc='upper left', fontsize=8)
    else:
        axes[-1].plot([], c='b', label='Brightness contour')
        axes[-1].legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="load model", default='../models/raft-synblobs.pth')
    parser.add_argument('--filename', help="input data file", default='../data/real_gpi/65472_0.35_processed.pbz2')
    parser.add_argument('--make_video', help="make video", action='store_true')
    parser.add_argument('--hand_labels', help="make video with hand labels", action='store_true')
    parser.add_argument('--viou_threshold', type=float, default=0.5, help="threshold for filtering prediction based on VIoU")
    parser.add_argument('--amp_threshold', type=float, default=0.75, help="threshold for filtering prediction based on blob amplitude")
    parser.add_argument('--blob_life_threshold', type=int, default=15, help="threshold for filtering prediction based on blob life span")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true', help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true', help='use position and content-wise attention')
    
    tick_all = time.time()
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_name = ''
    if 'raft' in args.model:
        parser.add_argument('--is_opticalFlow', default=True)
        parser.add_argument('--model_label', default='RAFT')
        model_name = 'raft'
        sys.path.append('motion/RAFT/core')
        from utils import flow_viz
        from utils.utils import InputPadder
        from raft import RAFT
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model, map_location=device))
        model = model.module
        def predict(model, image1, image2):
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            return flow_up
    elif 'gma' in args.model:
        parser.add_argument('--is_opticalFlow', default=True)
        parser.add_argument('--model_label', default='GMA')
        model_name = 'gma'
        sys.path.append('motion/GMA/core')
        from utils import flow_viz
        from utils.utils import InputPadder
        from network import RAFTGMA
        model = torch.nn.DataParallel(RAFTGMA(args))
        model.load_state_dict(torch.load(args.model, map_location=device))
        model = model.module
        def predict(model, image1, image2):
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            return flow_up
    elif 'mrcnn' in args.model:
        parser.add_argument('--is_opticalFlow', default=False)
        parser.add_argument('--model_label', default='Mask R-CNN')
        model_name = 'mrcnn'
        sys.path.append('motion/mask_rcnn/core')
        from model_data_prep import *
        model = get_model_instance_segmentation(3, dropout_prob=0.)
        model.load_state_dict(torch.load(args.model, map_location=device))
        def predict(model, image):
            result = model([image])[0]
            return result
    elif 'flowwalk' in args.model:
        parser.add_argument('--is_opticalFlow', default=True)
        parser.add_argument('--model_label', default='Flow Walk')
        model_name = 'flowwalk'
        sys.path.append('motion/flowwalk/core')
        from utils import flow_viz
        from utils.utils import InputPadder
        from Regressor import PWCLite
        from easydict import EasyDict
        cfg = EasyDict({"n_frames": 2, "reduce_dense": True, "type": "pwclite", "upsample": True})
        model = torch.nn.DataParallel(PWCLite(cfg))
        model.load_state_dict(torch.load(args.model, map_location=device))
        model = model.module
        def predict(model, image1, image2):
            im_all = torch.cat([image1, image2], 1)/255.
            flow_pred = model(im_all)['flows_fw']
            flow_predictions = [flow_pred[0]]
            flow_predictions[0][:,0,:,:] *= 255.
            return flow_predictions[0]
    
    args = parser.parse_args()
    print(f"Loaded checkpoint at {args.model}")
    model.to(device)
    model.eval()
    
    save_name_prefix = args.filename.split('_processed.pbz2')[0] + '_' + model_name
    if glob.glob(save_name_prefix + '.pickle'):
        with open(save_name_prefix + '.pickle', 'rb') as handle:
            data = pickle.load(handle)
        output = data['output']
        output_tracking = data['output_tracking']
        per_frame_pred_time = data['per_frame_pred_time']
        per_frame_proc_time = data['per_frame_proc_time']
    else:
        output, output_tracking, per_frame_pred_time, per_frame_proc_time = run_pred(args, model, device, predict)
        data = {'output':np.array(output), 'output_tracking':output_tracking, 'per_frame_pred_time':per_frame_pred_time, 'per_frame_proc_time':per_frame_proc_time}
        with open(save_name_prefix + '.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    tock_all = time.time()
    print(args.model_label + " Mean per-frame prediction time: " + str(per_frame_pred_time) + " sec")
    print(args.model_label + " Mean per-frame processing time: " + str(per_frame_proc_time) + " sec")
    print(args.model_label + " Total time elapsed (for tracking): " + str((tock_all - tick_all)/3600.) + " hours")
    viou_all = []
    for t in range(len(output_tracking)):
        for info in output_tracking[t]:
            _, viou, _, _, _, _ = info
            viou_all.append(viou)
    print(args.model_label + " Mean VIoU: " + str(np.mean(viou_all)))
    
    if args.make_video:
        tick_all = time.time()
        figs_pred, figs_fwhm, figs_id, figs_hl = [], [], [], []
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=7200)
        if args.is_opticalFlow:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
        else:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
        
        if args.hand_labels:
            hand_labels = []
            handlabel_files = sorted(glob.glob(args.filename.split('.pbz2')[0] + '_hand_label_*.pickle'))
            for file in handlabel_files:
                with open(file, 'rb') as handle:
                    hand_labels.append(pickle.load(handle))
            plot_args = (output, output_tracking, axes, args, hand_labels,)
        else:
            plot_args = (output, output_tracking, axes, args,)
        
        anim = animation.FuncAnimation(fig, plot_frame, init_func=init, fargs=plot_args, interval=50, frames=len(output))
        anim.save(save_name_prefix + '.mp4', writer=writer)
        tock_all = time.time()
        print(args.model_label + " Time elapsed for making video: " + str((tock_all - tick_all)/3600.) + " hours")



