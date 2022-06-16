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

def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 3 #Class 0: background, Class 1: region inside the shear layer, Class 2: blob
    mask_threshold = 0.95
    num_mc_passes = 100
    
    files = sorted(glob(osp.join('../data/synthetic_gpi', 'synthetic_gpi_*.pbz2')))
    dataset = []
    for file in files:
        with bz2.BZ2File(file, 'rb') as f:
            dataset.append(cPickle.load(f))
    
    data_aug_p_train = {"horizontalFlip":args.horizontalFlip, "scale":args.scale, "translate":args.translate, "rotate":args.rotate, "shear":args.shear}
    
    images_original, objects_original = process(device, dataset)
    split_list = np.loadtxt('../data/synthetic_gpi/synblobs_split.txt', dtype=np.int32)
    split_list = np.append(split_list, [1]*(len(objects_original) - len(split_list)))
    
    objects_original_train, objects_original_test = [], []
    for i in range(len(objects_original)):
        if split_list[i] == 1:
            objects_original_train.append(objects_original[i])
        elif split_list[i] == 2:
            objects_original_test.append(objects_original[i])
    
    images_aug, objects_aug = add_augmented_data(images_original[split_list == 1], objects_original_train, get_transform, data_aug_p_train, device)
    
    if len(images_aug) > 0:
        images_train = copy.deepcopy(np.concatenate([images_original[split_list == 1], images_aug], axis=0))
        objects_train = copy.deepcopy(objects_original_train + objects_aug)
    else:
        images_train = copy.deepcopy(images_original[split_list == 1])
        objects_train = copy.deepcopy(objects_original_train)
    
    images_test = copy.deepcopy(images_original[split_list == 2])
    if device.type == 'cpu':
        images_train = torch.from_numpy(images_train).type(torch.FloatTensor)
        images_test = torch.from_numpy(images_test).type(torch.FloatTensor)
    else:
        images_train = torch.from_numpy(images_train).type(torch.cuda.FloatTensor)
        images_test = torch.from_numpy(images_test).type(torch.cuda.FloatTensor)
    
    dataset_train = GPIDataset(images_train, objects_train)
    dataset_test = GPIDataset(images_test, objects_original_test)
    
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)
    
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, dropout_prob=args.dropout)
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
    
    if args.restore_ckpt is not None:
        checkpoint = torch.load(args.restore_ckpt, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler = checkpoint['lr_scheduler']
        avg_losses = checkpoint['avg_losses']
    else:
        start_epoch = 0
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        avg_losses = []
    
    for epoch in range(start_epoch, args.num_steps):
        # train for one epoch, printing every 10 iterations
        avg_loss, _ = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        avg_losses.append(avg_loss)
        
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        
        # update the learning rate
        lr_scheduler.step()
        checkpoint = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler, 'avg_losses':avg_losses}
        torch.save(checkpoint, args.output_ckpt + f'/{args.name}.pt')
    
    PATH = args.output_model + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

