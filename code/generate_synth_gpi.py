'''
Generates files of a synthetic blob moving along the trajectory, with full complexity.
'''
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import gaussian_filter
import bz2
import pickle
import _pickle as cPickle
import cv2 as cv
import random
import argparse
import os

def make_shear_layer(x1, y1, x2, y2, r, x, y):
    '''
    Makes a shear layer as an arc connecting given two points, with given radius.
    
    Inputs:
        x1, y1 (float): (x, y)-coordinates of a point in the range [0, 1], on the lower edge connecting the arc of the shear layer.
        x2, y2 (float): (x, y)-coordinates of a point in the range [0, 1], on the upper edge connecting the arc of the shear layer.
        r (float): radius of the arc of the shear layer.
        x, y (float): 2-D grid array of (x, y)-coordinates.
    
    Outputs:
        mask_inside_shear (float): 2-D grid array of mask which has 1's for indices inside the shear layer.
    
    (x0, y0) is the coordinate of the center of the circle, and their expressions in terms of x1, y1, x2, y2, r are retrieved from the following:
    #https://www.wolframalpha.com/input/?i=solve+%28x1-x2%2Bsqrt%28r%5E2-%28y2-y0%29%5E2%29%29%5E2+%2B+%28y1-y0%29%5E2+%3D+r%5E2+for+y0
    '''
    y0 = (np.sqrt((-4*x1**2*y1 - 4*x1**2*y2 + 8*x1*x2*y1 + 8*x1*x2*y2 - 4*x2**2*y1 - 4*x2**2*y2 - 4*y1**3 + 4*y1**2*y2 + 4*y1*y2**2 - 4*y2**3)**2 - 4*(4*x1**2 - 8*x1*x2 + 4*x2**2 + 4*y1**2 - 8*y1*y2 + 4*y2**2)*(-4*r**2*x1**2 + 8*r**2*x1*x2 - 4*r**2*x2**2 + x1**4 - 4*x1**3*x2 + 6*x1**2*x2**2 + 2*x1**2*y1**2 + 2*x1**2*y2**2 - 4*x1*x2**3 - 4*x1*x2*y1**2 - 4*x1*x2*y2**2 + x2**4 + 2*x2**2*y1**2 + 2*x2**2*y2**2 + y1**4 - 2*y1**2*y2**2 + y2**4)) + 4*x1**2*y1 + 4*x1**2*y2 - 8*x1*x2*y1 - 8*x1*x2*y2 + 4*x2**2*y1 + 4*x2**2*y2 + 4*y1**3 - 4*y1**2*y2 - 4*y1*y2**2 + 4*y2**3)/(2*(4*x1**2 - 8*x1*x2 + 4*x2**2 + 4*y1**2 - 8*y1*y2 + 4*y2**2))
    x0 = x2 - np.sqrt(r**2 - (y2 - y0)**2)
    mask_inside_shear = np.zeros((len(x), len(y)))
    for i, y_val in enumerate(y):
        mask_inside_shear[:, i] = np.where(x > x0 + np.sqrt(r**2 - (y_val - y0)**2), np.zeros(len(x)), 1.)
    
    return mask_inside_shear


class Flow:
    def __init__(self, t0, x, y, sx, sy, amp, angle, v_mag, v_ang):
        '''
        Constructs an elliptical flow.
        
        Inputs:
            t0 (int): the moment when this flow appears first.
            x (numpy array of int): trajectory of x-indices of the flow.
            y (numpy array of int): trajectory of y-indices of the flow.
            sx (numpy array of float): half-size of the flow along one of its axis throughout the trajectory.
            sy (numpy array of float): half-size of the flow along one of its axis throughout the trajectory.
            amp (numpy array of float): amplitude of the flow throughout the trajectory.
            angle (numpy array of float): tilt-angle of the axis of the flow throughout the trajectory.
        '''
        self.t0 = t0
        self.x = x
        self.y = y
        self.sx = sx
        self.sy = sy
        self.amp = amp
        self.angle = angle
        self.step = 0
        self.on_frame = True
        self.v_mag = v_mag
        self.v_ang = v_ang
    
    def evolve(self):
        '''
        Evolves to the next moment in the trajectory.
        '''
        if self.on_frame:
            if self.step + 1 < len(self.x):
                self.step += 1
            else:
                self.on_frame = False

def gaussian_2d(x, y, x0, y0, sx, sy, A):
    return A * np.exp(-(((x - x0) / sx)**2/2 + ((y - y0) / sy)**2/2))

def ellipse(x, y, x0, y0, sx, sy, A):
    return ((x - x0)*np.cos(A) + (y - y0)*np.sin(A))**2 / sx**2 + ((x - x0)*np.sin(A) - (y - y0)*np.cos(A))**2 / sy**2

def add_flows(t0s, flows, brt_grid, xind_grid, yind_grid, isblob, hsv, num_margin_xy):
    num_x, num_y, num_t = np.shape(brt_grid)
    mask = np.zeros((len(t0s) + 1, np.shape(brt_grid)[0], np.shape(brt_grid)[1], np.shape(brt_grid)[2])).astype(int)
    steps_frame = []
    for i, t0 in enumerate(t0s):
        flow = flows[i]
        t = t0
        steps = []
        while flow.on_frame and t < num_t - 1:
            xind_grid_rot = (xind_grid - flow.x[flow.step]) * np.cos(flow.angle[flow.step]) + (yind_grid - flow.y[flow.step]) * np.sin(flow.angle[flow.step])
            yind_grid_rot = -(xind_grid - flow.x[flow.step]) * np.sin(flow.angle[flow.step]) + (yind_grid - flow.y[flow.step]) * np.cos(flow.angle[flow.step])
            flowamp_gaus = gaussian_2d(xind_grid_rot, yind_grid_rot, 0, 0, flow.sx[flow.step], flow.sy[flow.step], flow.amp[flow.step])
            brt_grid[:, :, t] = np.maximum(brt_grid[:, :, t], flowamp_gaus)
            if isblob:
                #size_factor = np.max([0., 2. * flow.amp[flow.step] - 1.]) # is 1.0 when the amp is 1.0 and 0.0 when the amp is <=0.5.
                #size_factor = np.sqrt(2*np.log(2)) #FWHM
                size_factor = np.sqrt(2*np.log(2))*0.9
                mask[i+1, :, :, t] = (ellipse(xind_grid, yind_grid, flow.x[flow.step], flow.y[flow.step], size_factor*flow.sx[flow.step], size_factor*flow.sy[flow.step], flow.angle[flow.step]) <= 1.).astype(int)
                hsv[1, :, :, t][mask[i+1, :, :, t] == 1] = flow.v_ang[flow.step]
                hsv[0, :, :, t][mask[i+1, :, :, t] == 1] = flow.v_mag[flow.step]
                if num_margin_xy//2 <= flow.x[flow.step] <= num_x - 1 - num_margin_xy//2 and num_margin_xy//2 <= flow.y[flow.step] <= num_y - 1 - num_margin_xy//2:
                    steps.append(flow.step)
            else:
                hsv[1, :, :, t][flowamp_gaus > 0.6] = flow.v_ang[flow.step]
                hsv[0, :, :, t][flowamp_gaus > 0.6] = flow.v_mag[flow.step]
            
            flow.evolve()
            t += 1
        steps_frame.append(steps)
    return brt_grid, mask, steps_frame, hsv

TAG_CHAR = np.array([202021.25], np.float32)
def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    
    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv
    
    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def main(args):
    num_margin_xy = 300
    num_x = args.image_size[0] + num_margin_xy
    num_y = args.image_size[1] + num_margin_xy
    
    num_margin_t = 100
    num_t = min(args.n_frame, 200) + num_margin_t
    num_blob = min(args.n_frame, 200) // 60
    
    brt_grid = np.zeros((num_x, num_y, num_t))
    x = np.linspace(0., 1., num_x)
    y = np.linspace(0., 1., num_y)
    
    x1 = num_margin_xy / 2. / num_x + ((num_x - num_margin_xy) / num_x) * (0.1 + 0.5 * np.random.rand(1)[0])
    y1 = 0.
    x2 = x1 + 0.1 + 0.02 * (np.random.rand(1)[0] * 2. - 1.)
    y2 = 1.
    r = 10. + 0.1 * (np.random.rand(1)[0] * 2. - 1.)
    
    y_grid, x_grid = np.meshgrid(x, y)
    mask_inside_shear = make_shear_layer(x1, y1, x2, y2, r, x, y)
    shear_contour = np.argmin(mask_inside_shear, axis=0) - 1
    
    #Background flow inside of the shear layer
    num_plasma_flow_inside = num_t // 5
    t0s_inside = np.zeros(num_plasma_flow_inside, dtype=int)
    for i in range(num_plasma_flow_inside):
        t0s_inside[i] = int(i * num_t // num_plasma_flow_inside + np.random.randint(num_t // num_plasma_flow_inside))
    
    flow_inside = []
    dir_inside = np.random.randint(2) * 2. - 1.
    for t0_inside in t0s_inside:
        v0_inside = ((num_y - num_margin_xy) / 60.) * (1. + 0.1 * (np.random.rand(1)[0] * 2. - 1.))
        y0_inside = (num_y - 1) * (1 - int(dir_inside)) // 2
        width_inside = np.argwhere(mask_inside_shear[:, y0_inside] == 1.)[-1][0]
        x0_inside = int(width_inside - (width_inside - num_margin_xy//2) * 0.7 * np.random.rand(1)[0])
        sx0_inside = (num_x - num_margin_xy) / 6.
        sy0_inside = (num_y - num_margin_xy) / 2.
        amp0_inside = 1.0 * ((x0_inside - num_margin_xy//2) / (width_inside - num_margin_xy//2))
        traj_inside_x = shear_contour - shear_contour[y0_inside] + x0_inside
        traj_inside_y = np.arange(num_y)[np.argmax(traj_inside_x >= 0) :][::int(dir_inside)]
        traj_inside_x = traj_inside_x[np.argmax(traj_inside_x >= 0) :][::int(dir_inside)]
        x_inside = np.array([traj_inside_x[0]])
        y_inside = np.array([traj_inside_y[0]])
        step = 0
        while step + int(v0_inside) < len(traj_inside_x):
            step += int(v0_inside)
            x_inside = np.append(x_inside, traj_inside_x[step])
            y_inside = np.append(y_inside, traj_inside_y[step])
        
        dsa_inside = 0.05 #maximum percentage of the sinusoidal change in the size of the flow.
        dampa_inside = 0.05 #maximum percentage of the sinusoidal change in the amplitude of the flow.
        sx_inside = sx0_inside + dsa_inside * sx0_inside * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_inside)) * np.arange(len(x_inside)) + 2.*np.pi*np.random.rand(1)[0]) #Fluctuates 1 +- 0.2 times throughout the trajectory
        sy_inside = sy0_inside + dsa_inside * sy0_inside * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_inside)) * np.arange(len(x_inside)) + 2.*np.pi*np.random.rand(1)[0]) #Fluctuates 1 +- 0.2 times throughout the trajectory
        amp_inside = amp0_inside + dampa_inside * amp0_inside * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_inside)) * np.arange(len(x_inside))) #Fluctuates 1 +- 0.2 times throughout the trajectory
        angle_inside = np.zeros(len(x_inside))
        v_mag_inside = np.zeros(len(x_inside)) - 1.
        v_ang_inside = np.zeros(len(x_inside)) + (dir_inside*np.pi/2.)
        flow_inside.append(Flow(t0_inside, x_inside, y_inside, sx_inside, sy_inside, amp_inside, angle_inside, v_mag_inside, v_ang_inside))
    
    #Background flow outside of the shear layer
    #num_plasma_flow_outside = num_t // 5
    num_plasma_flow_outside = num_t // 3
    t0s_outside = np.zeros(num_plasma_flow_outside, dtype=int)
    for i in range(num_plasma_flow_outside):
        t0s_outside[i] = int(i * num_t // num_plasma_flow_outside + np.random.randint(num_t // num_plasma_flow_outside))
    
    flow_outside = []
    dir_outside = - dir_inside
    for t0_outside in t0s_outside:
        v0_outside = ((num_y - num_margin_xy) / 60.) * (1. + 0.1 * (np.random.rand(1)[0] * 2. - 1.))
        y0_outside = (num_y - 1) * (1 - int(dir_outside)) // 2
        width_inside = np.argwhere(mask_inside_shear[:, y0_outside] == 1.)[-1][0]
        x0_outside = int(width_inside + 1 + (num_x - 1 - num_margin_xy//2 - width_inside) * 0.5 * np.random.rand(1)[0])
        #sx0_outside = (num_x - num_margin_xy) / 6.
        #sy0_outside = (num_y - num_margin_xy) / 2.
        sx0_outside = (num_x - num_margin_xy) / 8.
        sy0_outside = (num_y - num_margin_xy) / 8.
        amp0_outside = 0.7 * (-x0_outside / (num_x - 1 - (width_inside + 1)) + (num_x - 1) / (num_x - 1 - (width_inside + 1)))
        traj_outside_x = shear_contour - shear_contour[y0_outside] + x0_outside
        traj_outside_y = np.arange(num_y)[np.argmax(traj_outside_x >= 0) :][::int(dir_outside)]
        traj_outside_x = traj_outside_x[np.argmax(traj_outside_x >= 0) :][::int(dir_outside)]
        x_outside = np.array([traj_outside_x[0]])
        y_outside = np.array([traj_outside_y[0]])
        step = 0
        while step + int(v0_outside) < len(traj_outside_x):
            step += int(v0_outside)
            x_outside = np.append(x_outside, traj_outside_x[step])
            y_outside = np.append(y_outside, traj_outside_y[step])
        
        dsa_outside = 0.05 #maximum percentage of the sinusoidal change in the size of the flow.
        dampa_outside = 0.05 #maximum percentage of the sinusoidal change in the amplitude of the flow.
        sx_outside = sx0_outside + dsa_outside * sx0_outside * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_outside)) * np.arange(len(x_outside)) + 2.*np.pi*np.random.rand(1)[0]) #Fluctuates 1 +- 0.2 times throughout the trajectory
        sy_outside = sy0_outside + dsa_outside * sy0_outside * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_outside)) * np.arange(len(x_outside)) + 2.*np.pi*np.random.rand(1)[0]) #Fluctuates 1 +- 0.2 times throughout the trajectory
        amp_outside = amp0_outside + dampa_outside * amp0_outside * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_outside)) * np.arange(len(x_outside))) #Fluctuates 1 +- 0.2 times throughout the trajectory
        angle_outside = np.zeros(len(x_outside))
        v_mag_outside = np.zeros(len(x_outside)) - 1.
        v_ang_outside = np.zeros(len(x_outside)) + (dir_outside*np.pi/2.)
        
        flow_outside.append(Flow(t0_outside, x_outside, y_outside, sx_outside, sy_outside, amp_outside, angle_outside, v_mag_outside, v_ang_outside))
    
    #Blobs
    t0s_blob = np.zeros(num_blob, dtype=int)
    for i in range(num_blob):
        t0s_blob[i] = int(num_margin_t + i * (num_t - num_margin_t) // num_blob + np.random.randint((num_t - num_margin_t) // num_blob // 8))
    
    blobs = []
    blob_types = []
    amp0_blob = 0.9 + 0.1 * np.random.rand(1)[0]
    for ib, t0_blob in enumerate(t0s_blob):
        y0_blob = (num_y - 1) * (1 - int(dir_outside)) // 2
        width_inside = np.argwhere(mask_inside_shear[:, y0_blob] == 1.)[-1][0]
        width_outside_blob = num_x - 1 - num_margin_xy//2 - width_inside
        x0_blob = width_inside + 1 + int(width_outside_blob * 0.5 * np.random.rand(1)[0])
        traj_base_x = shear_contour - shear_contour[y0_blob] + x0_blob
        traj_base_y = np.arange(num_y)[np.argmax(traj_base_x >= 0) :]
        traj_base_x = traj_base_x[np.argmax(traj_base_x >= 0) :]
        sx0_blob = (1.0 + (np.random.rand(1)[0] - 0.5) / 3.) * (num_x - num_margin_xy) / 6.
        sy0_blob = (1.0 + (np.random.rand(1)[0] - 0.5) / 3.) * (num_y - num_margin_xy) / 6.
        num_turning_pts = 1 + np.random.randint(2)
        blob_type = {'split':False, 'merge':False, 'appear':False, 'diminish':False}
        blob_type[list(blob_type.keys())[np.random.randint(4)]] = True
        for type, val in blob_type.items():
            if val:
                blob_types.append(type)
                break
        
        clipx_left = x0_blob + 1
        clipx_right = num_x - 1 - num_margin_xy//2
        clipy_left = num_margin_xy//2
        clipy_right = num_y - 1 - num_margin_xy//2
        if num_turning_pts == 1:
            mean_y = clipy_left + (clipy_right - clipy_left) / 2.
            sy = mean_y - clipy_left
            turning_pt_y = -1
            while not clipy_left < turning_pt_y < clipy_right:
                turning_pt_y = int(np.random.normal(mean_y, sy, 1)[0])
            
            clipx_left = np.max([clipx_left, traj_base_x[turning_pt_y]])
            mean_x = clipx_left + (clipx_right - clipx_left) / 2.
            sx = mean_x - clipx_left
            turning_pt_x = -1
            while not clipx_left <= turning_pt_x <= clipx_right-1:
                turning_pt_x = int(np.random.normal(mean_x, sx, 1)[0])
            
            turning_pts_x = np.array([turning_pt_x])
            turning_pts_y = np.array([turning_pt_y])
            
        elif num_turning_pts == 2:
            mean_y1 = clipy_left + (clipy_right - clipy_left) * 2. / 3.
            sy1 = clipy_right - mean_y1
            turning_pt1_y = -1
            while not clipy_left < turning_pt1_y < clipy_right:
                turning_pt1_y = int(np.random.normal(mean_y1, sy1, 1)[0])
            
            clipy_left = turning_pt1_y - (turning_pt1_y - clipy_left) * 1./ 2.
            mean_y2 = clipy_left + (turning_pt1_y - clipy_left) * 1./ 2.
            sy2 = mean_y2 - clipy_left
            turning_pt2_y = -1
            while not clipy_left < turning_pt2_y < turning_pt1_y:
                turning_pt2_y = int(np.random.normal(mean_y2, sy2, 1)[0])
            
            clipx_left = np.max([clipx_left, traj_base_x[turning_pt1_y]])
            mean_x1 = clipx_left + (clipx_right - clipx_left) / 3.
            sx1 = mean_x1 - clipx_left
            turning_pt1_x = -1
            while not clipx_left <= turning_pt1_x <= clipx_right-2:
                turning_pt1_x = int(np.random.normal(mean_x1, sx1, 1)[0])
            
            clipx_left = turning_pt1_x + (clipx_right - turning_pt1_x) / 5.
            mean_x2 = clipx_left + (clipx_right - clipx_left) / 2.
            sx2 = clipx_right - mean_x2
            turning_pt2_x = -1
            while not clipx_left < turning_pt2_x <= clipx_right-1:
                turning_pt2_x = int(np.random.normal(mean_x2, sx2, 1)[0])
            
            if turning_pt2_x - turning_pt1_x >= 5:
                turning_pts_x = np.array([turning_pt1_x, turning_pt2_x])
                turning_pts_y = np.array([turning_pt1_y, turning_pt2_y])[::int(dir_outside)]
            else:
                turning_pts_x = np.array([turning_pt1_x])
                turning_pts_y = np.array([turning_pt1_y])
                num_turning_pts = 1
        
        if dir_outside == 1.:
            min_angle = 10. * np.pi / 180.
            max_angle = 80. * np.pi / 180.
            endpoint_angle = min_angle + np.random.rand(1)[0] * (max_angle - min_angle)
            endpoint_y = np.min([num_y - 1, int(turning_pts_y[-1] + (num_x - 1 - turning_pts_x[-1]) * np.tan(endpoint_angle))])
            endpoint_x = np.min([num_x - 1, int(turning_pts_x[-1] + (num_y - 1 - turning_pts_y[-1]) / np.tan(endpoint_angle))])
            startpoint_dydx = (turning_pts_y[0] - y0_blob) / (turning_pts_x[0] - x0_blob)
            if blob_type['split']:
                endpoint_angle_s = min_angle + np.random.rand(1)[0] * (max_angle - min_angle)
                endpoint_y_s = np.min([num_y - 1, int(turning_pts_y[-1] + (num_x - 1 - turning_pts_x[-1]) * np.tan(endpoint_angle_s))])
                endpoint_x_s = np.min([num_x - 1, int(turning_pts_x[-1] + (num_y - 1 - turning_pts_y[-1]) / np.tan(endpoint_angle_s))])
        else:
            min_angle = -80. * np.pi / 180.
            max_angle = -10. * np.pi / 180.
            endpoint_angle = min_angle + np.random.rand(1)[0] * (max_angle - min_angle)
            endpoint_y = np.max([0, int(turning_pts_y[-1] + (num_x - 1 - turning_pts_x[-1]) * np.tan(endpoint_angle))])
            endpoint_x = np.min([num_x - 1, int(turning_pts_x[-1] - turning_pts_y[-1] / np.tan(endpoint_angle))])
            startpoint_dydx = (turning_pts_y[0] - y0_blob) / (turning_pts_x[0] - x0_blob)
            if blob_type['split']:
                endpoint_angle_s = min_angle + np.random.rand(1)[0] * (max_angle - min_angle)
                endpoint_y_s = np.max([0, int(turning_pts_y[-1] + (num_x - 1 - turning_pts_x[-1]) * np.tan(endpoint_angle_s))])
                endpoint_x_s = np.min([num_x - 1, int(turning_pts_x[-1] - turning_pts_y[-1] / np.tan(endpoint_angle_s))])
        
        if blob_type['merge']:
            if num_turning_pts == 1:
                x0_blob_m = width_inside + 1 + int((turning_pts_x[-1] - width_inside - 2) * np.random.rand(1)[0])
            elif num_turning_pts == 2:
                x0_blob_m = x0_blob + 1 + int((turning_pts_x[-1] - x0_blob - 2) * np.random.rand(1)[0])
            y0_blob_m = y0_blob
            startpoint_dydx_m = (turning_pts_y[-1] - y0_blob_m) / (turning_pts_x[-1] - x0_blob_m)
        
        traj_pts_x = np.hstack([[x0_blob], turning_pts_x, [endpoint_x]])
        traj_pts_y = np.hstack([[y0_blob], turning_pts_y, [endpoint_y]])
        traj_pts_dydx = np.hstack([[startpoint_dydx], turning_pts_x * 0., [np.tan(endpoint_angle)]])
        trajyf = CubicHermiteSpline(x=traj_pts_x, y=traj_pts_y, dydx=traj_pts_dydx)
        angle0_blob = np.random.rand(1)[0] * np.pi #initial tilt-angle of the axis of the blob.
        
        v_blob_max = (1. + 0.05 * (np.random.rand(1)[0] - 0.5)) * (num_y - num_margin_xy) / 50.
        if num_turning_pts == 1:
            traj_pts_steps = traj_pts_x - x0_blob
            traj_pts_speeds = np.array([1., 0.3, 1.])
        elif num_turning_pts == 2:
            traj_pts_steps = np.hstack([[x0_blob], [turning_pts_x[0], np.mean(turning_pts_x), turning_pts_x[1]], endpoint_x]) - x0_blob
            traj_pts_speeds = np.array([1., 0.3, 0.35, 0.3, 1.])
        
        vf = CubicHermiteSpline(x=traj_pts_steps, y=traj_pts_speeds, dydx=np.zeros_like(traj_pts_steps))
        
        def get_v_traj(vf, v_max, trajyf, x0, y0, xf):
            x = np.array([x0])
            y = np.array([y0])
            v_mag = []
            v_ang = []
            while True:
                dt = 1
                dx = 1
                dy = trajyf(x[-1] + dx) - trajyf(x[-1])
                vx = v_max * vf(x[-1] - x0) * dx / np.sqrt(dx**2 + dy**2)
                vy = v_max * vf(x[-1] - x0) * dy / np.sqrt(dx**2 + dy**2)
                x = np.append(x, x[-1] + vx*dt)
                y = np.append(y, y[-1] + vy*dt)
                v_mag.append(np.sqrt(vx**2 + vy**2))
                v_ang.append(np.arctan(vy/vx))
                if int(x[-1]) > xf or not 0 <= int(y[-1]) <= num_y-1:
                    x = x[:-1]
                    y = y[:-1]
                    v_mag = v_mag[:-1]
                    v_ang = v_ang[:-1]
                    break
            
            v_mag.append(v_mag[-1])
            v_ang.append(v_ang[-1])
            return x.astype(int), y.astype(int), np.array(v_mag), np.array(v_ang)
        
        x_blob, y_blob, v_mag, v_ang = get_v_traj(vf, v_blob_max, trajyf, x0_blob, y0_blob, endpoint_x)
        
        dsa_blob = 0.3 #maximum percentage of the sinusoidal change in the size of the blob.
        dampa_blob = 0.05 #maximum percentage of the sinusoidal change in the amplitude of the blob.
        dangle_blob = 0.1 * np.pi #the change of tilt-angle of the blob throughout the trajectory.
        sx_blob = sx0_blob + dsa_blob * sx0_blob * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_blob)) * np.arange(len(x_blob)) + 2.*np.pi*np.random.rand(1)[0]) #Fluctuates 1 +- 0.2 times throughout the trajectory
        sy_blob = sy0_blob + dsa_blob * sy0_blob * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_blob)) * np.arange(len(x_blob)) + 2.*np.pi*np.random.rand(1)[0]) #Fluctuates 1 +- 0.2 times throughout the trajectory
        amp_blob = amp0_blob + dampa_blob * amp0_blob * np.sin(2. * np.pi * ((1. + 0.4*np.random.rand(1)[0]-0.2) / len(x_blob)) * np.arange(len(x_blob))) #Fluctuates 1 +- 0.2 times throughout the trajectory
        angle_blob = np.linspace(angle0_blob, angle0_blob + dir_outside * dangle_blob, len(x_blob)) #Rotates clockwise (for the blob moving down) or counter-clockwise (for the blob moving up) by dangle radian throughout the trajectory
        if blob_type['appear']:
            amp_blob = np.append(np.linspace(0.51, amp0_blob, len(x_blob)//2), np.repeat(amp0_blob, len(x_blob) - len(x_blob)//2))
            sx_blob = np.append(np.linspace(0.3, 1.0, len(sx_blob)//2), np.repeat(1.0, len(sx_blob) - len(sx_blob)//2)) * sx_blob
            sy_blob = np.append(np.linspace(0.3, 1.0, len(sy_blob)//2), np.repeat(1.0, len(sy_blob) - len(sy_blob)//2)) * sy_blob
        elif blob_type['diminish']:
            amp_blob = np.append(np.repeat(amp0_blob, len(x_blob) - len(x_blob)//2), np.linspace(amp0_blob, 0.51, len(x_blob)//2))
            sx_blob = np.append(np.linspace(1.0, 0.3, len(sx_blob)//2), np.repeat(0.3, len(sx_blob) - len(sx_blob)//2)) * sx_blob
            sy_blob = np.append(np.linspace(1.0, 0.3, len(sy_blob)//2), np.repeat(0.3, len(sy_blob) - len(sy_blob)//2)) * sy_blob
        
        blobs.append(Flow(t0_blob, x_blob, y_blob, sx_blob, sy_blob, amp_blob, angle_blob, v_mag, v_ang))
        
        if blob_type['split']:
            for t in range(len(x_blob)):
                if x_blob[t] > turning_pts_x[-1]:
                    if (dir_outside == 1. and y_blob[t] > turning_pts_y[-1]) or (dir_outside == -1. and y_blob[t] < turning_pts_y[-1]):
                        startpoint_x_s = x_blob[t]
                        startpoint_y_s = y_blob[t]
                        sx0_blob_s = blobs[-1].sx[t]
                        sy0_blob_s = blobs[-1].sy[t]
                        amp0_blob_s = blobs[-1].amp[t]
                        angle0_blob_s = blobs[-1].angle[t]
                        t0_blob_s = t0_blob + t
                        traj_perc = (t + 1.) / len(x_blob)
                        break
            
            traj_pts_x_s = np.hstack([startpoint_x_s, [endpoint_x_s]])
            traj_pts_y_s = np.hstack([startpoint_y_s, [endpoint_y_s]])
            traj_pts_dydx_s = np.hstack([[0.], [np.tan(endpoint_angle_s)]])
            trajyf_s = CubicHermiteSpline(x=traj_pts_x_s, y=traj_pts_y_s, dydx=traj_pts_dydx_s)
            traj_pts_steps_s = traj_pts_x_s - startpoint_x_s
            traj_pts_speeds_s = np.array([0.3, 1.])
            vf_s = CubicHermiteSpline(x=traj_pts_steps_s, y=traj_pts_speeds_s, dydx=np.zeros_like(traj_pts_steps_s))
            x_blob_s, y_blob_s, v_mag_s, v_ang_s = get_v_traj(vf_s, v_blob_max, trajyf_s, startpoint_x_s, startpoint_y_s, endpoint_x_s)
            
            sx_blob_s = sx0_blob_s + dsa_blob * sx0_blob * np.sin(2. * np.pi * ((1. - traj_perc) / len(x_blob_s)) * np.arange(len(x_blob_s)))
            sy_blob_s = sy0_blob_s + dsa_blob * sy0_blob * np.sin(2. * np.pi * ((1. - traj_perc) / len(x_blob_s)) * np.arange(len(x_blob_s)))
            amp_blob_s = amp0_blob_s + dampa_blob * amp0_blob * np.sin(2. * np.pi * ((1. - traj_perc) / len(x_blob_s)) * np.arange(len(x_blob_s)))
            angle_blob_s = np.linspace(angle0_blob_s, angle0_blob_s + dir_outside * dangle_blob * (1. - traj_perc), len(x_blob_s))
            blobs.append(Flow(t0_blob_s, x_blob_s, y_blob_s, sx_blob_s, sy_blob_s, amp_blob_s, angle_blob_s, v_mag_s, v_ang_s))
        elif blob_type['merge']:
            for t in range(len(x_blob)):
                if x_blob[t] >= turning_pts_x[-1]:
                    if (dir_outside == 1. and y_blob[t] >= turning_pts_y[-1]) or (dir_outside == -1. and y_blob[t] <= turning_pts_y[-1]):
                        endpoint_x_m = x_blob[t-1]
                        endpoint_y_m = y_blob[t-1]
                        sxf_blob_m = blobs[-1].sx[t-1]
                        syf_blob_m = blobs[-1].sy[t-1]
                        ampf_blob_m = blobs[-1].amp[t-1]
                        anglef_blob_m = blobs[-1].angle[t-1]
                        tf_blob_m = t0_blob + t - 1
                        traj_perc = t / len(x_blob)
                        break
            
            traj_pts_x_m = np.hstack([[x0_blob_m], endpoint_x_m])
            traj_pts_y_m = np.hstack([[y0_blob_m], endpoint_y_m])
            traj_pts_dydx_m = np.hstack([[startpoint_dydx_m], [0.]])
            trajyf_m = CubicHermiteSpline(x=traj_pts_x_m, y=traj_pts_y_m, dydx=traj_pts_dydx_m)
            traj_pts_steps_m = traj_pts_x_m - x0_blob_m
            traj_pts_speeds_m = np.array([1., 0.3])
            vf_m = CubicHermiteSpline(x=traj_pts_steps_m, y=traj_pts_speeds_m, dydx=np.zeros_like(traj_pts_steps_m))
            x_blob_m, y_blob_m, v_mag_m, v_ang_m = get_v_traj(vf_m, v_blob_max, trajyf_m, x0_blob_m, y0_blob_m, endpoint_x_m)
            
            t0_blob_m = tf_blob_m - len(x_blob_m) + 1
            sx_blob_m = np.flip(sxf_blob_m + dsa_blob * sx0_blob * np.sin(2. * np.pi * (traj_perc / len(x_blob_m)) * np.arange(len(x_blob_m))))
            sy_blob_m = np.flip(syf_blob_m + dsa_blob * sy0_blob * np.sin(2. * np.pi * (traj_perc / len(x_blob_m)) * np.arange(len(x_blob_m))))
            amp_blob_m = np.flip(ampf_blob_m + dampa_blob * amp0_blob * np.sin(2. * np.pi * (traj_perc / len(x_blob_m)) * np.arange(len(x_blob_m))))
            angle_blob_m = np.flip(np.linspace(anglef_blob_m, anglef_blob_m - dir_outside * dangle_blob * traj_perc, len(x_blob_m)))
            blobs.append(Flow(t0_blob_m, x_blob_m, y_blob_m, sx_blob_m, sy_blob_m, amp_blob_m, angle_blob_m, v_mag_m, v_ang_m))
    
    t0s_blob = []
    i_in_frame = []
    for i in range(len(blobs)):
        if blobs[i].t0 < num_t:
            t0s_blob.append(blobs[i].t0)
            i_in_frame.append(i)
    
    blobs = [blobs[i] for i in i_in_frame]
    yind_grid, xind_grid = np.meshgrid(np.arange(num_x), np.arange(num_y))
    hsv = np.zeros((2, np.shape(brt_grid)[0], np.shape(brt_grid)[1], np.shape(brt_grid)[2]))
    brt_grid, _, _, hsv = add_flows(t0s_inside, flow_inside, brt_grid, xind_grid, yind_grid, False, hsv, num_margin_xy)
    brt_grid, _, _, hsv = add_flows(t0s_outside, flow_outside, brt_grid, xind_grid, yind_grid, False, hsv, num_margin_xy)
    
    hsv = np.zeros((2, np.shape(brt_grid)[0], np.shape(brt_grid)[1], np.shape(brt_grid)[2]))
    
    brt_grid, blob_mask, steps_frame, hsv = add_flows(t0s_blob, blobs, brt_grid, xind_grid, yind_grid, True, hsv, num_margin_xy)
    blob_mask[0, :, :, :] = np.repeat(mask_inside_shear[:, :, np.newaxis].astype(int), np.shape(blob_mask)[3], axis=2)
    hsv[0, :, :, :][hsv[0, :, :, :] == -1.] = np.max(hsv[0, :, :, :])*0.1
    hsv[0, :, :, :] = hsv[0, :, :, :] * 255. / np.max(hsv[0, :, :, :])
    hsv[0, :, :, :][hsv[0, :, :, :] > 255.] = 255.
    
    brt_true = brt_grid[num_margin_xy//2 : num_x - num_margin_xy//2, num_margin_xy//2 : num_y - num_margin_xy//2, num_margin_t:]
    brt_true = gaussian_filter(brt_true, sigma=[15, 15, 0])
    brt_true /= np.max(brt_true)
    blob_mask = blob_mask[:, num_margin_xy//2 : num_x - num_margin_xy//2, num_margin_xy//2 : num_y - num_margin_xy//2, num_margin_t:]
    hsv = hsv[:, num_margin_xy//2 : num_x - num_margin_xy//2, num_margin_xy//2 : num_y - num_margin_xy//2, num_margin_t:]
    
    x_idx = np.linspace(0, np.shape(brt_true)[0] - 1, 12).astype(int)
    y_idx = np.linspace(0, np.shape(brt_true)[1] - 1, 10).astype(int)
    brt_downsampled = np.zeros((12, 10, np.shape(brt_true)[2]))
    for i in range(12):
        for j in range(10):
            brt_downsampled[i, j] = np.round(brt_true[x_idx[i], y_idx[j], :], 3)
    
    shear_contour_x = shear_contour[num_margin_xy//2 : num_x - num_margin_xy//2] - num_margin_xy//2
    shear_contour_y = np.arange(0, num_x - num_margin_xy)
    data = {'brt_true':brt_true, 'brt_downsampled':brt_downsampled, 'shear_contour_x':shear_contour_x, 'shear_contour_y':shear_contour_y, 'blob_mask':blob_mask, 'blob_type':blob_types, 'hsv':hsv}
    with bz2.BZ2File(args.output + '/synthetic_gpi_' + "{:03d}".format(n+1) + '.pbz2', 'w') as f:
        cPickle.dump(data, f)
    
    for t in range(np.shape(hsv)[3] - 1):
        vx = hsv[0, :, :, t] * np.cos(hsv[1, :, :, t])
        vy = hsv[0, :, :, t] * np.sin(hsv[1, :, :, t])
        writeFlow(args.output + '/' + "{:05d}".format((num_t-num_margin_t)*n + t) + '_flow.flo', vx, vy)
        cv.imwrite(args.output + '/' + "{:05d}".format((num_t-num_margin_t)*n + t) + '_img1.png', 255.*(1. - brt_true[:, :, t]))
        cv.imwrite(args.output + '/' + "{:05d}".format((num_t-num_margin_t)*n + t) + '_img2.png', 255.*(1. - brt_true[:, :, t+1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--n_frame', type=int, help="Number of frames in a file (>= 200)", default=200)
    parser.add_argument('--n_data', type=int, help="Number of data files to generate", default=30)
    parser.add_argument('--val_prop', type=int, help="Proportion of total frames for validation data", default=0.05)
    parser.add_argument('--output', type=str, help="output directory to save data", default='../data/synthetic_gpi')
    args = parser.parse_args()
    #Output data "brt_true" size: (num_x - num_margin_xy) X (num_y - num_margin_xy) X (num_t - num_margin_t)
    #Output data "brt_downsampled" size: 12 X 10 X (num_t - num_margin_t)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    
    for n in range(args.n_data):
        main(args)
    
    num_all = args.n_data*(max(args.n_frame, 200) - 1)
    idx_val = random.sample(range(num_all), int(num_all*args.val_prop))
    idx = [1]*num_all
    for i in idx_val:
        idx[i] = 2
    
    with open(args.output + '/synblobs_split.txt', 'w') as f:
        for item in idx:
            f.write("%s\n" % item)
