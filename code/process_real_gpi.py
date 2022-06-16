import argparse
import numpy as np
import bz2
import pickle
import _pickle as cPickle
from scipy.interpolate import Rbf

def upsample(args, brt_arr, r_arr, z_arr):
    n_x, n_y = args.image_size
    n_t = np.shape(brt_arr)[2]
    brt_upsampled = np.zeros((n_x, n_y, n_t))
    y_grid, x_grid = np.meshgrid(np.linspace(np.min(z_arr), np.max(z_arr), n_y), np.linspace(np.min(r_arr), np.max(r_arr), n_x))
    for t in range(n_t):
        idx_nan = np.isnan(brt_arr[:,:,t])
        brt_upsampled[:,:,t] = Rbf(r_arr[~idx_nan], z_arr[~idx_nan], brt_arr[:,:,t][~idx_nan], function='cubic')(x_grid, y_grid)
    
    return brt_upsampled

def normalize_brt(brt_true, idx_shear_x, idx_shear_y):
    mean_brt_arr = np.repeat(np.mean(brt_true, axis=2)[:, :, np.newaxis], np.shape(brt_true)[2], axis=2)
    std_view = np.std(brt_true, axis=2)
    brt_true = (brt_true - mean_brt_arr) / np.repeat(std_view[:, :, np.newaxis], np.shape(brt_true)[2], axis=2)
    
    n_x, n_y, n_t = np.shape(brt_true)
    dx = int(n_x*0.05)
    dy = int(n_y*0.05)
    max_brt_outside = np.array([])
    min_brt_outside = np.array([])
    for i in range(dy, len(idx_shear_y)-dy):
        max_brt_outside = np.append(max_brt_outside, np.max(brt_true[dx:-dx, dy:-dy, :][idx_shear_x[i]-dx:, idx_shear_y[i]-dy, :], axis=1))
        min_brt_outside = np.append(min_brt_outside, np.min(brt_true[dx:-dx, dy:-dy, :][idx_shear_x[i]-dx:, idx_shear_y[i]-dy, :], axis=1))
    
    brt_upper_cap = np.mean(max_brt_outside)
    brt_lower_cap = np.mean(min_brt_outside)
    
    brt_true = np.minimum(brt_true, brt_upper_cap)
    brt_true = np.maximum(brt_true, brt_lower_cap)
    brt_true -= np.min(brt_true)
    brt_true /= np.max(brt_true)
    
    return brt_true

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help="input data file", default='../data/real_gpi/65472_0.35_raw.pickle')
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256])
    args = parser.parse_args()
    
    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
    
    brt_arr = data['brt_arr']
    r_arr = data['r_arr']
    z_arr = data['z_arr']
    shear_contour_x = data['shear_contour_x']
    shear_contour_y = data['shear_contour_y']
    
    brt_true = upsample(args, brt_arr, r_arr, z_arr)
    brt_true = normalize_brt(brt_true, shear_contour_x, shear_contour_y)
    
    output = {'brt_true':brt_true, 'shear_contour_x':shear_contour_x, 'shear_contour_y':shear_contour_y}
    with bz2.BZ2File(args.filename.split('_raw')[0] + '_processed.pbz2', 'w') as f:
        cPickle.dump(output, f)
