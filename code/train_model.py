import argparse
import sys
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-synblobs', help="name your experiment")
    parser.add_argument('--stage', default='synblobs', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, default='synblobs', nargs='+')
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=0.0001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    
    parser.add_argument('--output', type=str, default='trained_models/checkpoints', help='output directory to save checkpoints and plots')
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')
    parser.add_argument('--model_name', default='', help='specify model name')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    
    #Parameters only for Mask R-CNN
    parser.add_argument('--momentum', type=float, default=0.2956)
    parser.add_argument('--step_size', type=int, default=12)
    parser.add_argument('--horizontalFlip', type=float, default=0.3413)
    parser.add_argument('--scale', type=float, default=0.5597)
    parser.add_argument('--translate', type=float, default=0.2614)
    parser.add_argument('--rotate', type=float, default=0.3679)
    parser.add_argument('--shear', type=float, default=0.5644)
    
    args = parser.parse_args()
    
    if not os.path.isdir('trained_models'):
        os.mkdir('trained_models')
    if not os.path.isdir('trained_models/checkpoints'):
        os.mkdir('trained_models/checkpoints')
    
    if 'raft' in args.name:
        sys.path.append('models/RAFT')
    elif 'gma' in args.name:
        sys.path.append('models/GMA')
    elif 'mrcnn' in args.name:
        sys.path.append('models/mask_rcnn')
    
    from train import main
    main(args)
