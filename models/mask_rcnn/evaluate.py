import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from model_data_prep import *
import numpy as np
import torch
import bz2
import pickle
import _pickle as cPickle
import copy
import os.path as osp
from glob import glob

def get_eval_model(model, dropout_prob):
    model.eval()
    for i, layer in enumerate(model.roi_heads.mask_head):
        if layer.__class__.__name__ == 'Sequential':
            model.roi_heads.mask_head[i][1].p = dropout_prob
            model.roi_heads.mask_head[i][1].train()
    return model

SMOOTH = 1e-6
def iou_numpy(output: np.array, label: np.array, img):
    output = output.squeeze(1)
    intersection = np.sum(img[0,:,:][(output[0,:,:] == 1) & (label[0,:,:] == 1)])
    union = np.sum(img[0,:,:][(output[0,:,:] == 1) | (label[0,:,:] == 1)])
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def get_ious(dataset, model, device, mask_threshold):
    ious = {1:[], 2:[]}
    for data in dataset:
        result = model([data[0].to(device)])[0]
        if len(result['labels']) == 0:
            continue
        for class_idx in [1, 2]:
            outputs = result['masks'][result['labels'] == class_idx].cpu().detach().numpy()
            outputs[outputs >= mask_threshold] = 1
            outputs[outputs < mask_threshold] = 0
            outputs = outputs.astype(int)
            if len(outputs) == 0 or np.max(outputs) == 0:
                continue
            labels = data[1]['masks'][data[1]['labels'] == class_idx].cpu().detach().numpy()
            num_output = np.shape(outputs)[0]
            num_label = np.shape(labels)[0]
            cost = np.zeros((num_output, num_label))
            for output_idx in range(num_output):
                for label_idx in range(num_label):
                    cost[output_idx, label_idx] = -iou_numpy(outputs[output_idx:output_idx+1], labels[label_idx:label_idx+1], data[0].cpu().detach().numpy())
            output_indices, label_indices = linear_sum_assignment(cost)
            if len(output_indices) > 0:
                iou = np.mean([-cost[output_indices[i], label_indices[i]] for i in range(len(output_indices))])
                ious[class_idx].append(iou)
    for class_idx in [1, 2]:
        if len(ious[class_idx]) == 0:
            ious[class_idx] = [0.]
    return ious

def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 3 #Class 0: background, Class 1: region inside the shear layer, Class 2: blob
    mask_threshold = 0.95
    num_mc_passes = 100
    
    files = sorted(glob(osp.join('data/synthetic_gpi', 'synthetic_gpi_*.pbz2')))
    dataset = []
    for file in files:
        with bz2.BZ2File(file, 'rb') as f:
            dataset.append(cPickle.load(f))
    
    images_original, objects_original = process(device, dataset)
    split_list = np.loadtxt('data/synthetic_gpi/synblobs_split.txt', dtype=np.int32)
    split_list = np.append(split_list, [1]*(len(objects_original) - len(split_list)))
    
    images_test = images_original[split_list == 2]
    objects_test = []
    for i in range(len(objects_original)):
        if split_list[i] == 2:
            objects_test.append(objects_original[i])
    
    if device.type == 'cpu':
        images_test = torch.from_numpy(images_test).type(torch.FloatTensor)
    else:
        images_test = torch.from_numpy(images_test).type(torch.cuda.FloatTensor)
    
    dataset_test = GPIDataset(images_test, objects_test)
    
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, dropout_prob=args.dropout)
    model.load_state_dict(torch.load(args.model, map_location=device))
    # move model to the right device
    model.to(device)
    
    model = get_eval_model(model, dropout_prob=0.30)
    ious_1 = []
    ious_2 = []
    for n in range(num_mc_passes):
        ious = get_ious(dataset_test, model, device, mask_threshold)
        ious_1 += ious[1]
        ious_2 += ious[2]
    
    print("Mean SHEAR IoU: " + str(np.mean(ious_1)))
    print("Std SHEAR IoU: " + str(np.mean(ious_2)))
    print("Mean BLOB IoU: " + str(np.std(ious_1)))
    print("Std BLOB IoU: " + str(np.std(ious_2)))

