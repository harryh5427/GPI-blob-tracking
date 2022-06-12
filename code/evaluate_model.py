import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="load model", default='../models/raft-synblobs.pth')
    parser.add_argument('--dataset', default='synblobs', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true', help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true', help='use position and content-wise attention')
    # Ablations
    parser.add_argument('--replace', default=False, action='store_true', help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true', help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true', help='Remove residual connection. Do not add local features with the aggregated features.')
    
    args = parser.parse_args()
    
    if 'raft' in args.model:
        sys.path.append('motion/RAFT')
    elif 'gma' in args.model:
        sys.path.append('motion/GMA')
    elif 'mrcnn' in args.model:
        sys.path.append('motion/mask_rcnn')
    
    from evaluate import *
    
    main(args)

