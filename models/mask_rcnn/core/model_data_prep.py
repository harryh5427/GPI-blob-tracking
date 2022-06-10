'''
Script for implementing the Mask R-CNN and training using synthetic blob dataset, following the tutorial in (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
'''
import numpy as np
import torch
import torchvision
import random

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from engine import train_one_epoch, evaluate
import utils
from data_aug.data_aug import *

from scipy.interpolate import Rbf
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import copy
from scipy.optimize import linear_sum_assignment


def process(device, dataset):
    '''
    Processes the dataset into the appropriate form.
    
    Inputs:
        dataset (list): the list of data from files to be processed and stored.
    Outputs:
        images (tensor): the tensor of the input images for the model. Size: (number of datapoints) X (3 RGB) X (number of y) X (number of x)
        objects (dict): contains keys as follows:
            'boxes': the list (tensor) of (x1, y1, x2, y2) of the bounding box of the objects. Here, x1 < x2 and y1 < y2.
            'labels': the list (tensor) of labels of the objects
            'masks': the list (tensor) of masks of the objects, where value 1 for the region of the object and 0 otherwise.
            'image_id': the list (tensor) of IDs of the image.
            'area': the list (tensor) of the area of the objects.
            'iscrowd': the list (tensor) of the number of objects.
    '''
    
    images = []
    objects = []
    #Iterate over the data files, and the data will be concatenated at the end.
    for i_data, data in enumerate(dataset):
        brt_true = data['brt_true'] #Size: (number of x) X (number of y) X (number of t)
        n_x, n_y, n_t = np.shape(brt_true)
        blob_mask = data['blob_mask'] #Size: (number of objects) X (number of x) X (number of y) X (number of t)
        
        images_i = []
        #Iterate over the frames, and the data will be concatenated at the end.
        for idx, t in enumerate(range(n_t)):
            #The class label 1 is for the region inside of the shear layer.
            x1, x2 = 0, np.max(data['shear_contour_x'])
            y1, y2 = np.min(data['shear_contour_y']), np.max(data['shear_contour_y'])
            boxes = np.array([[x1, y1, x2, y2]])
            labels = [1]
            num_objs = 1
            obj_idx = [0]
            blob_mask_t = blob_mask[:,:,:,t]
            
            #The frame is transposed and prepended with two additional dimensions, to match with the desired input size (number of datapoints) X (3 RGB) X (number of y) X (number of x)
            brt_true_3chan = np.repeat(brt_true[:,:,t].T[np.newaxis, np.newaxis, :, :], 3, axis=1)
            images_i = np.concatenate([images_i, brt_true_3chan], axis=0) if len(images_i) > 0 else brt_true_3chan
            
            idx_redundant_all = []
            idx_redundant_pairs = []
            for i_blob in range(1, np.shape(blob_mask_t)[0]):
                i_blob_redundant = []
                for j_blob in range(1, np.shape(blob_mask_t)[0]):
                    if i_blob != j_blob and i_blob not in idx_redundant_all and np.sum(blob_mask_t[i_blob,:,:]-blob_mask_t[j_blob,:,:]) == 0:
                        i_blob_redundant.append(j_blob)
                if len(i_blob_redundant) > 0:
                    idx_redundant_all += [i_blob] + i_blob_redundant
                    idx_redundant_pairs += i_blob_redundant
            redundant_filter = [True]*np.shape(blob_mask_t)[0]
            for idx_red in idx_redundant_pairs:
                redundant_filter[idx_red] = False
            blob_mask_t = blob_mask_t[redundant_filter,:,:]
            
            for i_blob in range(1, np.shape(blob_mask_t)[0]):
                idx_x, idx_y = np.where(blob_mask_t[i_blob,:,:] == 1.)
                if len(idx_x) > 0:
                    labels.append(2)
                    num_objs += 1
                    obj_idx.append(i_blob)
                    x1, y1, x2, y2 = np.min(idx_x), np.min(idx_y), np.max(idx_x), np.max(idx_y)
                    #Takes care of the case when x1 == x2 or y1 == y2
                    if x1 == x2:
                        if x2 == n_x:
                            x1 = x2 - 1
                        else:
                            x2 = x1 + 1
                    if y1 == y2:
                        if y2 == n_y:
                            y1 = y2 - 1
                        else:
                            y2 = y1 + 1
                    if np.min(boxes) == -1:
                        boxes = np.array([[x1, y1, x2, y2]])
                    else:
                        boxes = np.vstack((boxes, [x1, y1, x2, y2]))
            
            if device.type == 'cpu':
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64).type(torch.LongTensor)
                boxes = torch.tensor(boxes).type(torch.FloatTensor)
                labels = torch.tensor(labels).type(torch.LongTensor)
                masks = torch.as_tensor(np.transpose(blob_mask_t[obj_idx][:,:,:], (0, 2, 1)), dtype=torch.uint8).type(torch.ByteTensor)
                image_id = torch.tensor([idx]).type(torch.LongTensor)
            else:
                iscrowd = torch.zeros((num_objs,), dtype=torch.int64).type(torch.cuda.LongTensor)
                boxes = torch.tensor(boxes).type(torch.cuda.FloatTensor)
                labels = torch.tensor(labels).type(torch.cuda.LongTensor)
                masks = torch.as_tensor(np.transpose(blob_mask_t[obj_idx][:,:,:], (0, 2, 1)), dtype=torch.uint8).type(torch.cuda.ByteTensor)
                image_id = torch.tensor([idx]).type(torch.cuda.LongTensor)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            objects.append({'boxes':boxes, 'labels':labels, 'masks':masks, 'image_id':image_id, 'area':area, 'iscrowd':iscrowd})
        
        images = np.concatenate([images, images_i], axis=0) if len(images) > 0 else images_i
    
    return images, objects

class GPIDataset(Dataset):
    def __init__(self, images, objects):
        '''
        Dataset class for the synthetic blob data.
        
        Inputs:
            images (tensor): the tensor of the input imagees for the model. Size: (number of datapoints) X (3 RGB) X (number of y) X (number of x)
            objects (dict): contains keys as follows:
                'boxes': the list (tensor) of (x1, y1, x2, y2) of the bounding box of the objects. Here, x1 < x2 and y1 < y2.
                'labels': the list (tensor) of labels of the objects
                'masks': the list (tensor) of masks of the objects, where value 1 for the region of the object and 0 otherwise.
                'image_id': the list (tensor) of IDs of the image.
                'area': the list (tensor) of the area of the objects.
                'iscrowd': the list (tensor) of the number of objects.
        '''
        self.images = images
        self.objects = objects
    
    def __getitem__(self, idx):
        image = self.images[idx,:,:,:]
        object = self.objects[idx]
        return copy.deepcopy(image), copy.deepcopy(object)
    
    def __len__(self):
        return self.images.shape[0]

def get_model_instance_segmentation(num_classes, dropout_prob):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.95)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    for layer in model.backbone.body:
        if layer == 'relu':
            model.backbone.body[layer] = torch.nn.Sequential(torch.nn.LeakyReLU(inplace=True), torch.nn.Dropout2d(p=0.))
        elif 'layer' in layer:
            for sublayer in model.backbone.body[layer]:
                sublayer.relu = torch.nn.Sequential(torch.nn.LeakyReLU(inplace=True), torch.nn.Dropout2d(p=0.))
    
    for i, layer in enumerate(model.roi_heads.mask_head):
        if layer.__class__.__name__ == 'ReLU':
            model.roi_heads.mask_head[i] = torch.nn.Sequential(torch.nn.LeakyReLU(inplace=True), torch.nn.Dropout2d(p=dropout_prob))
    
    return model

def get_transform(image, object, data_aug_p, device):
    image_t = np.transpose(copy.deepcopy(image), (1,2,0))
    boxes_t = copy.deepcopy(object['boxes'].cpu().numpy())
    masks_t = np.transpose(copy.deepcopy(object['masks'].cpu().numpy()), (1,2,0))
    masks_t_temp = np.zeros((np.shape(masks_t)[0], np.shape(masks_t)[1], np.shape(masks_t)[2]+1))
    masks_t_temp[:,:,:-1] = masks_t
    masks_t = masks_t_temp
    labels_t = copy.deepcopy(object['labels'].cpu().numpy())
    area_t = copy.deepcopy(object['area'].cpu().numpy())
    iscrowd_t = copy.deepcopy(object['iscrowd'].cpu().numpy())
    is_transformed = False
    
    def correct_boxes(boxes, n_x, n_y):
        boxes = boxes.astype(int).astype(float)
        boxes = np.maximum(boxes, 0.)
        boxes[:,2] = np.minimum(boxes[:,2], n_x-1)
        boxes[:,3] = np.minimum(boxes[:,3], n_y-1)
        for i_box in range(np.shape(boxes)[0]):
            x1, y1, x2, y2 = boxes[i_box]
            #Takes care of the case when x1 == x2 or y1 == y2
            if x1 == x2:
                if x2 == n_x:
                    x1 = x2 - 1
                else:
                    x2 = x1 + 1
            if y1 == y2:
                if y2 == n_y:
                    y1 = y2 - 1
                else:
                    y2 = y1 + 1
            boxes[i_box] = np.array([x1, y1, x2, y2])
        return boxes
    
    class apply_horizontalFlip():
        def __call__(self, image, boxes, masks):
            image_t = copy.deepcopy(image)
            boxes_t = copy.deepcopy(boxes)
            masks_t = copy.deepcopy(masks)
            transform_f = RandomHorizontalFlip(1.0)
            image_t, boxes_t = transform_f(copy.deepcopy(image), copy.deepcopy(boxes))
            if transform_f.flipped:
                masks_t, boxes_t = HorizontalFlip()(copy.deepcopy(masks), copy.deepcopy(boxes))
            boxes_t = correct_boxes(copy.deepcopy(boxes_t), np.shape(image_t)[1], np.shape(image_t)[0])
            return image_t, boxes_t, masks_t
    
    class apply_scale():
        def __call__(self, image, boxes, masks):
            image_t = copy.deepcopy(image)
            boxes_t = copy.deepcopy(boxes)
            masks_t = copy.deepcopy(masks)
            transform_f = RandomScale(0.5, diff=True)
            image_t, boxes_t = transform_f(copy.deepcopy(image), copy.deepcopy(boxes))
            masks_t, boxes_t = Scale(scale_x=transform_f.scale_x, scale_y=transform_f.scale_y)(copy.deepcopy(masks), copy.deepcopy(boxes))
            boxes_t = correct_boxes(copy.deepcopy(boxes_t), np.shape(image_t)[1], np.shape(image_t)[0])
            return image_t, boxes_t, masks_t
    
    class apply_translate():
        def __call__(self, image, boxes, masks):
            image_t = copy.deepcopy(image)
            boxes_t = copy.deepcopy(boxes)
            masks_t = copy.deepcopy(masks)
            transform_f = RandomTranslate(translate=(0.05,0.2), diff = True)
            image_t, boxes_t = transform_f(copy.deepcopy(image), copy.deepcopy(boxes))
            masks_t, boxes_t = Translate(translate_x=transform_f.translate_factor_x, translate_y=transform_f.translate_factor_y)(copy.deepcopy(masks), copy.deepcopy(boxes))
            boxes_t = correct_boxes(copy.deepcopy(boxes_t), np.shape(image_t)[1], np.shape(image_t)[0])
            return image_t, boxes_t, masks_t
    
    class apply_rotate():
        def __call__(self, image, boxes, masks):
            image_t = copy.deepcopy(image)
            boxes_t = copy.deepcopy(boxes)
            masks_t = copy.deepcopy(masks)
            transform_f = RandomRotate(angle=10)
            image_t, boxes_t = transform_f(copy.deepcopy(image), copy.deepcopy(boxes))
            masks_t, boxes_t = Rotate(transform_f.angle_chosen)(copy.deepcopy(masks), copy.deepcopy(boxes))
            boxes_t = correct_boxes(copy.deepcopy(boxes_t), np.shape(image_t)[1], np.shape(image_t)[0])
            return image_t, boxes_t, masks_t
    
    class apply_shear():
        def __call__(self, image, boxes, masks):
            image_t = copy.deepcopy(image)
            boxes_t = copy.deepcopy(boxes)
            masks_t = copy.deepcopy(masks)
            transform_f = RandomShear(shear_factor=0.2)
            image_t, boxes_t = transform_f(copy.deepcopy(image), copy.deepcopy(boxes))
            masks_t, boxes_t = Shear(shear_factor=transform_f.shear_factor_chosen)(copy.deepcopy(masks), copy.deepcopy(boxes))
            boxes_t = correct_boxes(copy.deepcopy(boxes_t), np.shape(image_t)[1], np.shape(image_t)[0])
            return image_t, boxes_t, masks_t
    
    transforms = [(apply_horizontalFlip(), data_aug_p["horizontalFlip"]), (apply_scale(), data_aug_p["scale"]), (apply_translate(), data_aug_p["translate"]), (apply_rotate(), data_aug_p["rotate"]), (apply_shear(), data_aug_p["shear"])]
    
    for i in range(len(transforms)):
        if random.random() < transforms[i][1]:
            num_err = 0
            while True:
                try:
                    image_temp, boxes_temp, masks_temp = transforms[i][0](image_t, boxes_t, masks_t)
                    box_okay = True
                    for box in boxes_temp:
                        x1, y1, x2, y2 = box
                        if x1 >= x2 or y1 >= y2:
                            box_okay = False
                    if box_okay:
                        image_t = copy.deepcopy(image_temp)
                        boxes_t = copy.deepcopy(boxes_temp)
                        masks_t = copy.deepcopy(masks_temp)
                        is_transformed = True
                        break
                except:
                    num_err += 1
                    if num_err == 10:
                        break
    
    masks_t = masks_t[:,:,:-1]
    
    if np.shape(boxes_t)[0] < np.shape(masks_t)[2]:
        box_match_idx = []
        box_match_score = []
        for i in range(np.shape(masks_t)[2]):
            num_ones = []
            for j in range(np.shape(boxes_t)[0]):
                num_ones.append(np.sum(masks_t[int(boxes_t[j,1]):int(boxes_t[j,3])+1, int(boxes_t[j,0]):int(boxes_t[j,2])+1, i]))
            if len(num_ones) > 0:
                box_match_idx.append(np.argmax(num_ones))
                box_match_score.append(np.max(num_ones))
        mask_idx_match = np.argsort(box_match_score)[::-1][:np.shape(boxes_t)[0]]
        mask_filter = [False]*np.shape(masks_t)[2]
        for idx in mask_idx_match:
            mask_filter[idx] = True
        masks_t = masks_t[:,:,mask_filter]
        labels_t = labels_t[mask_filter]
        area_t = area_t[mask_filter]
        iscrowd_t = iscrowd_t[mask_filter]
    
    mask_filter = [True]*np.shape(masks_t)[2]
    for i in range(np.shape(masks_t)[2]):
        if np.max(masks_t[:,:,i]) == 0:
            mask_filter[i] = False
    masks_t = masks_t[:,:,mask_filter]
    boxes_t = boxes_t[mask_filter,:]
    labels_t = labels_t[mask_filter]
    area_t = area_t[mask_filter]
    iscrowd_t = iscrowd_t[mask_filter]
    
    mask_filter = [True]*np.shape(masks_t)[2]
    for i_blob in range(np.shape(masks_t)[2]):
        idx_y, idx_x = np.where(masks_t[:,:,i_blob] == 1.)
        if len(idx_x) > 0:
            x1, y1, x2, y2 = np.min(idx_x), np.min(idx_y), np.max(idx_x), np.max(idx_y)
            #Takes care of the case when x1 == x2 or y1 == y2
            if x1 == x2:
                if x2 == np.shape(masks_t)[1] - 1:
                    x1 = x2 - 1
                else:
                    x2 = x1 + 1
            if y1 == y2:
                if y2 == np.shape(masks_t)[0] - 1:
                    y1 = y2 - 1
                else:
                    y2 = y1 + 1
            boxes_t[i_blob] = np.array([x1, y1, x2, y2])
        else:
            mask_filter[i_blob] = False
    masks_t = masks_t[:,:,mask_filter]
    boxes_t = boxes_t[mask_filter,:]
    labels_t = labels_t[mask_filter]
    area_t = area_t[mask_filter]
    iscrowd_t = iscrowd_t[mask_filter]
    
    object_t = copy.deepcopy(object)
    if device.type == 'cpu':
        object_t['boxes'] = torch.from_numpy(boxes_t.astype(int).astype(float)).type(torch.FloatTensor)
        object_t['masks'] = torch.from_numpy(np.transpose(masks_t, (2,0,1)).astype(int)).type(torch.ByteTensor)
        object_t['labels'] = torch.from_numpy(labels_t).type(torch.LongTensor)
        object_t['area'] = torch.from_numpy(area_t).type(torch.FloatTensor)
        object_t['iscrowd'] = torch.from_numpy(iscrowd_t).type(torch.LongTensor)
    else:
        object_t['boxes'] = torch.from_numpy(boxes_t.astype(int).astype(float)).type(torch.cuda.FloatTensor)
        object_t['masks'] = torch.from_numpy(np.transpose(masks_t, (2,0,1)).astype(int)).type(torch.cuda.ByteTensor)
        object_t['labels'] = torch.from_numpy(labels_t).type(torch.cuda.LongTensor)
        object_t['area'] = torch.from_numpy(area_t).type(torch.cuda.FloatTensor)
        object_t['iscrowd'] = torch.from_numpy(iscrowd_t).type(torch.cuda.LongTensor)
    return is_transformed, copy.deepcopy(np.transpose(image_t, (2,0,1))), object_t

def add_augmented_data(images, objects, transforms, data_aug_p, device):
    images_aug = []
    objects_aug = []
    for i in range(len(images)):
        is_transformed, image_aug, object_aug = transforms(images[i], objects[i], data_aug_p, device)
        if is_transformed:
            image_aug_3chan = np.repeat(image_aug[np.newaxis, :, :, :], 1, axis=0)
            images_aug = np.concatenate([images_aug, image_aug_3chan], axis=0) if len(images_aug) > 0 else image_aug_3chan
            objects_aug.append(object_aug)
    
    return images_aug, objects_aug
