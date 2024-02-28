# Copyright (C) 2023 OPPO. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import imageio
import numpy as np
import argparse
import os
import sys
cpwd = os.getcwd()
sys.path.append(cpwd)
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R_func
from tqdm import tqdm
from src.load_llfff import load_llff_data


import psutil

def calculate_angle1(x, y):
    angle = np.arctan2(x, y)
    angle = np.where(angle < 0, angle + 2*np.pi, angle)
    return (((2*np.pi - angle) - np.pi)*2)/np.pi
def calculate_angle2(x, y):
    angle = np.arctan2(x, y)
    angle = np.where(angle < 0, angle + 2*np.pi, angle)
    return ((angle - np.pi)*2)/np.pi


def direction_to_euler(vec):
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    theta = calculate_angle1(y,z)
    phi = calculate_angle1(x,z)

    return theta, phi

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str, default = 'dataset/Ollie/',help = 'exp name') #data/llff/nelf/house2/
parser.add_argument("--factor", type=int, default=4, help='downsample factor for LLFF images')
parser.add_argument("--only_renderpose", action='store_true' ,default = False)


if __name__ == "__main__":

    print("1018 new vec3 xyz! new train")
    process = psutil.Process(os.getpid())
    print('start : ',process.memory_info().rss / (1024 ** 2), "MB")
    # process the pose
    args = parser.parse_args()
    images, poses, bds, render_poses, i_test,focal_depth = load_llff_data(args.data_dir, args.factor,
                                                                recenter=False, bd_factor=1)
    data_dir  = args.data_dir
    print("Load image : ",process.memory_info().rss / (1024 ** 2), "MB")

    testskip=101
    val_idx = np.asarray([k for k in range(poses.shape[0]) if k%testskip==0])
    train_idx = np.asarray([k for k in range(poses.shape[0])])

    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape,  hwf, args.data_dir)
    
    def save_idx_rgb(idx, label):
        rgb_path = f"{data_dir}/rgb.npy"
        rgb = np.reshape(images[idx],(-1,3))
        print("rgb = ", rgb.shape)
        np.save(rgb_path.replace('.npy', f'{label}.npy'), rgb)

    save_idx_rgb(train_idx, 'train')
    save_idx_rgb(val_idx, 'val')

    images = None
    print("image = None : ",process.memory_info().rss / (1024 ** 2), "MB")

    uvst_path = f"{data_dir}/uvst.npy"
   


    # Cast intrinsics to correct types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]])

    print(f'val_idx {val_idx} train_idx {train_idx}')
   

    

    def save_idx(idx, label):

        
        
        uvst_tmp = []
        for p in poses[idx,:3,:4]:
            
            aspect = W/H
            
            u = np.linspace(-1, 1, W, dtype='float32')
            v = np.linspace(1, -1, H, dtype='float32') / aspect
            vu = list(np.meshgrid(u, v))

            u = vu[0]
            v = vu[1] 
            dirs = np.stack((u, v, -np.ones_like(u)), axis=-1)
            dirs = np.sum(dirs[..., np.newaxis, :]* p[:3,:3],-1)
            dirs = np.array(dirs)
            dirs = np.reshape(dirs,(-1,3))
            
            x = np.ones_like(vu[0]) * p[0,3]
            y = np.ones_like(vu[0]) * p[1,3] 
            z = np.ones_like(vu[0]) * p[2,3] 
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            x = np.reshape(x,(-1,1))
            y = np.reshape(y,(-1,1))
            z = np.reshape(z,(-1,1))

    
            concatenated_array = np.concatenate((dirs, x, y, z), axis=1)
            uvst_tmp.append(concatenated_array)

        uvst_tmp = np.asarray(uvst_tmp)
        uvst_tmp = np.reshape(uvst_tmp,(-1,6))
        
        
        print("uvst = ", uvst_tmp.shape)
       
        np.save(uvst_path.replace('.npy', f'{label}.npy'), uvst_tmp)
        
        print("npy generate finish : ",process.memory_info().rss / (1024 ** 2), "MB")


    print("npy generate start : ",process.memory_info().rss / (1024 ** 2), "MB")

    save_idx(train_idx, 'train')
    save_idx(val_idx, 'val')
