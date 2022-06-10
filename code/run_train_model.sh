#!/bin/bash
python train_model.py --name raft-synblobs --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 256 256 --wdecay 0.0001
#python train_model.py --name gma-synblobs --gpus 0 1 --num_steps 120000 --batch_size 6 --lr 0.000125 --image_size 256 256 --wdecay 0.0001 --mixed_precision
#python train_model.py --name mrcnn-synblobs --lr 0.05 --momentum 0.2956 --wdecay 2.729e-05 --num_steps 28 --step_size 12 --gamma 0.402 --batch_size 2
