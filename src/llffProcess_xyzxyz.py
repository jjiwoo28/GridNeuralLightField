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
from src.cam_view import rayPlaneInter
from src.utils import get_rays_np

def makedirs(path):
    if os.path.exists(path):
        return
    else:
        os.makedirs(path)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str, default = 'dataset/Ollie/',help = 'exp name') #data/llff/nelf/house2/
parser.add_argument("--factor", type=int, default=4, help='downsample factor for LLFF images')
parser.add_argument("--grid_n", type=int, default=2)

if __name__ == "__main__":
    testskip=60

    args = parser.parse_args()
    data_dir  = args.data_dir
    paths = []
    train_path =os.path.join(data_dir,"train")
    val_path =os.path.join(data_dir,"val")
    paths.append(train_path)
    paths.append(val_path)

    for p in paths:
        makedirs(p)
    


    
    
    

    # arr_test1 = np.load(v_path_array[3]) 
    # arr_test2 = np.load(h_path_array[3]) 
    # arr_stack = np.stack((arr_test1,arr_test2),axis = -1)


    # breakpoint()
    # process the pose
#    images, poses, bds, render_poses, i_test,focal_depth = load_llff_data(args.data_dir, args.factor,
#                                                                recenter=True, bd_factor=.75)
    images, poses, bds, render_poses, i_test,focal_depth = load_llff_data(args.data_dir, args.factor,
                                                                recenter=True, bd_factor=None)



    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape,  hwf, args.data_dir)

    # Cast intrinsics to correct types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    K = np.array([
    [focal, 0, 0.5*W],
    [0, focal, 0.5*H],
    [0, 0, 1]])

    # Backup code
    # val_idx = np.asarray([k for k in range(poses.shape[0]) if k%testskip==0])
    # train_idx = np.asarray([k for k in range(poses.shape[0]) if k%testskip])
    # print(f'val_idx {val_idx} train_idx {train_idx}')

    testskip=101
    val_idx = np.asarray([k for k in range(poses.shape[0]) if k%testskip==0])
    

  
    # Specific index
    val_idx = [72, 76, 80, 140, 144, 148, 208, 212, 216]
    train_idx = [x for x in range(poses.shape[0])]
    for ii in range(0, len(val_idx)):
        train_idx.remove(val_idx[ii])
    print(f'val_idx {val_idx} train_idx {train_idx}')

    val_idx = np.asarray(val_idx)
    train_idx = np.asarray(train_idx)

    


    def gen_6d_light_field(i,type):
        
        p = poses[i,:3,:4]
        aspect = W/H
        aspect = W/H
            
        u = np.linspace(-1, 1, W, dtype='float32')
        v = np.linspace(1, -1, H, dtype='float32') / aspect
        vu = list(np.meshgrid(u, v))

        u = vu[0]
        v = vu[1] 
        dirs = np.stack((u, v, -np.ones_like(u)), axis=-1)
        dirs = np.sum(dirs[..., np.newaxis, :]* p[:3,:3],-1)
        dirs = np.array(dirs)
        
            
        x = np.ones_like(vu[0]) * p[0,3]
        y = np.ones_like(vu[0]) * p[1,3] 
        z = np.ones_like(vu[0]) * p[2,3] 
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        x = x[:, :, np.newaxis]
        y = y[:, :, np.newaxis]
        z = z[:, :, np.newaxis]

        xyzxyz = np.concatenate((dirs, x, y, z), axis=-1)

        base_path= os.path.join(data_dir,type)
        uvst_path = os.path.join(base_path,"xyzxyz")
        makedirs(uvst_path)
        np.save(os.path.join(uvst_path,f"{i:04}.npy"), xyzxyz)

    
    

    # save_idx_with_HV(train_idx, 'train')    
    # save_idx_with_HV(val_idx, 'val')
    # interset radius plane 
    def gen_uvst_meta(i, type):
        p = poses[i,:3,:4]
        aspect = W/H
        u = np.linspace(-1, 1, W, dtype='float32')
        v = np.linspace(1, -1, H, dtype='float32') / aspect

        vu = list(np.meshgrid(u, v))

        u = vu[0]
        v = vu[1]
        s = np.ones_like(vu[0]) * p[0, 3]
        t = np.ones_like(vu[1]) * p[1, 3]
        uvst =np.stack((u, v, s, t), axis=-1)
        
        base_path= os.path.join(data_dir,type)
        uvst_path = os.path.join(base_path,"uvst")
        makedirs(uvst_path)
        np.save(os.path.join(uvst_path,f"{i:04}.npy"), uvst)


    def gen_rgb(i,type):

        rgb = np.array(images[i])
        

       

        base_path= os.path.join(data_dir,type)
        uvst_path = os.path.join(base_path,"rgb")
        makedirs(uvst_path)
        np.save(os.path.join(uvst_path,f"{i:04}.npy"), rgb)


    def gen_yuv(i, type):
        # images[i]를 YUV로 변환 (images는 RGB 이미지를 포함하는 배열이라고 가정)
        rgb = np.array(images[i])
        bgr_img = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    
        # BGR 이미지를 YUV 컬러 스페이스로 변환
        yuv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)

        
        # 결과를 저장할 경로 설정
        base_path = os.path.join(data_dir, type)
        uvst_path = os.path.join(base_path, "yuv")
        if not os.path.exists(uvst_path):
            os.makedirs(uvst_path)

        # 변환된 YUV 데이터를 .npy 파일로 저장
        np.save(os.path.join(uvst_path, f"{i:04}.npy"), yuv_img)
        
        
    for i in train_idx:
         gen_6d_light_field(i, "train")
         gen_rgb(i, "train")
         #gen_yuv(i, "train")

    for i in val_idx:
         gen_6d_light_field(i, "val")
         gen_rgb(i, "val")
         #gen_yuv(i, "val")
        

    
    # for i in range(n):
    #     for j in range(n):
    #         save_idx_grid(train_idx ,'train',i,j,n)
    #         save_idx_grid(val_idx ,'val',i,j,n)
    #         print(f"{i} + {j} + @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")    
            
    # save_idx(train_idx, 'train')
    # save_idx(val_idx, 'val')
