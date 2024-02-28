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

import sys
import os
import logging
from datetime import datetime
from line_profiler import LineProfiler

import cProfile
import pstats




sys.path.insert(0, os.path.abspath('./'))
from numpy.lib.stride_tricks import broadcast_to

from src.load_llfff import normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import argparse
import cv2
import matplotlib.pyplot as plt
import re
from PIL import Image





from torch.utils.tensorboard import SummaryWriter

from src.multi_model import GridNetworks
import torchvision
import glob
import torch.optim as optim
from utils import eval_uvst,rm_folder,AverageMeter,rm_folder_keep,eval_trans
from src.tracker import ResultTracker 
from mytimer import Timer

from src.cam_view import rayPlaneInter,get_rays,rotation_matrix_from_vectors

import imageio
from src.utils import get_rays_np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips



parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--data_dir',type=str, 
                    default = 'dataset/Ollie/',help='data folder name')
parser.add_argument('--batch_size',type=int, default = 8192,help='normalize input')
parser.add_argument('--test_freq',type=int,default=10,help='test frequency')
parser.add_argument('--val_freq',type=int,default=10,help='test frequency')
parser.add_argument('--save_checkpoints',type=int,default=10,help='checkpoint frequency')
parser.add_argument('--whole_epoch',type=int,default=300,help='checkpoint frequency')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument("--factor", type=int, default=4, help='downsample factor for LLFF images')
parser.add_argument('--img_scale',type=int, default= 1, help= "devide the image by factor of image scale")
parser.add_argument('--norm_fac',type=float, default=1, help= "normalize the data uvst")
parser.add_argument('--st_norm_fac',type=float, default=1, help= "normalize the data uvst")
parser.add_argument('--work_num',type=int, default= 15, help= "normalize the data uvst")
parser.add_argument('--lr_pluser',type=int, default = 100,help = 'scale for dir')
parser.add_argument('--lr',type=float,default=5e-04,help='learning rate')
parser.add_argument('--loadcheckpoints', action='store_true', default = False)


parser.add_argument('--use_6D', action='store_true', default = False)
parser.add_argument('--render_only', action='store_true', default = False)
parser.add_argument('--use_yuv', action='store_true', default = False)
parser.add_argument('--st_depth',type=int, default= 0, help= "st depth")
parser.add_argument('--uv_depth',type=int, default= 0.0, help= "uv depth")
parser.add_argument('--rep',type=int, default=1)
parser.add_argument('--mlp_depth', type=int, default = 4)
parser.add_argument('--mlp_width', type=int, default = 64)
parser.add_argument('--load_epoch', type=int, default = 0)

parser.add_argument('--grid_n', type=int, default = 2)
parser.add_argument('--loss_clamp', action='store_true', default = False)
parser.add_argument('--loss_clamp_value', type=int , default=5)


class PsnrSaver:
    # 클래스의 정적 속성 정의
    psnr = []
    label = []
    result_path = None




def plot_psnr_3d_with_labels(data, labels, file_path):
    """
    3차원 배열과 라벨 배열을 받아서 PSNR 그래프를 그립니다.
    :param data: (label의 수, 샘플링할 epoch의 수, 2) 형태의 배열. 각 요소는 (epoch, psnr) 형태
    :param labels: 각 데이터 세트(라인)에 해당하는 라벨의 이름을 담은 배열
    """
    data = np.array(data)
    labels = np.array(labels)
    # data.shape[0]는 라벨의 개수
    for i in range(data.shape[0]):
        epochs = data[i, :, 0]  # 첫 번째 차원의 값은 에폭
        psnr_values = data[i, :, 1]  # 두 번째 차원의 값은 PSNR 값들

        # 그래프에 라인 추가
        plt.plot(epochs, psnr_values, label=labels[i])

    plt.xlabel('Epoch')
    plt.ylabel('PSNR Mean')
    plt.title('PSNR Mean Over Time')
    plt.grid(True)
    plt.legend()

    # 그래프를 파일로 저장
    rm_folder_keep(file_path)
    path = os.path.join(file_path,"result.png")
    plt.savefig(path)

    return plt

class train():
    def __init__(self,args , mode = "uvst"):
         # set gpu id
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))
        print("240205_888")
    
        self.mode = mode
        # data root
        
        self.data_dir = args.data_dir
        n = args.grid_n

        print(f"grid_n = {args.grid_n}  !!!!!!!!!!!!!!!!!!!!!!!!!")
        self.grid_n = args.grid_n
        
        self.grig_n = args.grid_n 
        self.batch_size = args.batch_size
        self.use_yuv = args.use_yuv

        self.use_6D = args.use_6D

        data_img = os.path.join(args.data_dir,'images_{}'.format(args.factor)) 
 
        self.exp = f"{args.exp_name}_grid{self.grid_n}_d{args.mlp_depth}w{args.mlp_width}" 

        #json process
        # self.result_json_path = f"result_json/grid_n-{args.grid_n},d-{args.mlp_depth},w-{args.mlp_width}.json"
        # self.result_tracker = ResultTracker(args.grid_n ,self.result_json_path)
        # self.result_tracker.load()

        # self.loss_clamp_value  = args.loss_clamp_value
        
        # PsnrSaver.result_path = os.path.join("result",args.exp_name + "PsnrResult")

        if self.use_6D:
            self.input_dim = 6
        else:
            self.input_dim = 4
        #model setting

        self.model = GridNetworks( grid_size= args.grid_n,D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width,input_dim=self.input_dim)
        #print(self.model)

        #path setting


        self.summary_writer = SummaryWriter("src/tensorboard/"+self.exp)
        self.img_folder_train = 'result/'+self.exp+'/train/'
        self.img_folder_test = 'result/'+self.exp+'/test/'
        self.checkpoints = 'result/'+self.exp+'/checkpoints/'
        
        rm_folder_keep(self.img_folder_train)
        rm_folder_keep(self.img_folder_test)
        rm_folder_keep(self.checkpoints)
        #log setting
        
        if(args.loadcheckpoints or args.render_only):
            if args.load_epoch == 0:
                self.load_check_points_multi()
            else:
                self.load_check_points_multi_select(args.load_epoch)

        self.model = self.model.cuda()
        
        handlers = [logging.StreamHandler()]
        dt_string = self.exp +"_"  + datetime.now().strftime("%d-%m-%Y-%H-%M")
        handlers.append(logging.FileHandler('result/'+self.exp+f'/{dt_string}.log', mode='w'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-5s %(message)s',
            datefmt='%m-%d %H:%M:%S', handlers=handlers,
        )
        
        

        

        # save img
        self.save_img = True


        # height and width
        image_paths = glob.glob(f"{data_img}/*.png")
        sample_img = cv2.imread(image_paths[0])
        self.h = int((sample_img.shape[0]/args.img_scale))
        self.w = int((sample_img.shape[1]/args.img_scale))

        self.img_scale = args.img_scale

        self.step_count = 0

        # load nelf data
        print(f"Start loading...")


        if(args.render_only):

            save_dir = self.img_folder_train + f"render-{self.iter}.mp4"
            
            rm_folder_keep(save_dir)
            if args.use_6D:
                uvsts = self.get_render_pose_6D()
            else:
                uvsts = self.get_render_pose()
            self.render_uvst(uvsts ,save_dir)


            sys.exit()
        
    


    def train_step(self,args):


        if self.use_6D:
            self.uvst_train = self.load_data_grid("xyzxyz" , "train",self.grid_n)
            self.uvst_val = self.load_data_grid("xyzxyz" , "val",self.grid_n)
            print("load xyzxyz")
        else:
            self.uvst_train = self.load_data_grid("uvst" , "train",self.grid_n)
            self.uvst_val = self.load_data_grid("uvst" , "val",self.grid_n)

        if args.use_yuv:
            self.rgb_train = self.load_data_grid("yuv" , "train",self.grid_n)
            self.rgb_val = self.load_data("yuv" , "val")
        else:
            self.rgb_train = self.load_data_grid("rgb" , "train",self.grid_n)
            self.rgb_val = self.load_data("rgb" , "val")
        #self.usst_test = self.load_data("uvst","train")

        self.val_size = self.rgb_val.shape[0]

       
        
        # for network_index, (network, optimizer) in enumerate(zip(self.model.networks, self.model.optimizers)):
        #     print(network_index)
        #     grid_v = network_index % self.grid_n
        #     grid_h = network_index // self.grid_n
        #     print(f"grid_v = {grid_v} , grid_h = {grid_h}")

        
        print("test")

       
        # data loader

     
        self.start, self.end = [], []
        s = 0
        self.split_whole_size = np.reshape(self.uvst_train[0, 0,:,:,:,:],(-1,self.input_dim)).shape[0]
        if self.split_whole_size < self.batch_size:
            self.batch_size = self.split_whole_size
        while s < self.split_whole_size:
            self.start.append(s)
            s += args.batch_size
            self.end.append(min(s, self.split_whole_size))
       
        # optimizer
       
        self.vis_step = 1
        
       
        self.epoch_num = args.whole_epoch
        start_epoch = 0
        if(args.loadcheckpoints):
                    start_epoch = self.iter

        Timer.start()

        for epoch in range(start_epoch,start_epoch+self.epoch_num+1):
            #print(f"epoch = {epoch}")
            self.losses = AverageMeter()
            for network_index, (network, optimizer , scheduler) in enumerate(zip(self.model.networks, self.model.optimizers , self.model.schedulers)):
                #print(f"network_index = {network_index}")
                grid_v = network_index % self.grid_n
                grid_h = network_index // self.grid_n

                self.uvst_train_gpu = torch.tensor(np.reshape(self.uvst_train[grid_h, grid_v,:,:,:,:],(-1,self.input_dim))).float()
                self.rgb_train_gpu = torch.tensor(np.reshape(self.rgb_train[grid_h, grid_v,:,:,:,:],(-1,3))).float()

                #print(f"network_index = {network_index} , cpu 2 gpu")

                 
                self.losses_batch = AverageMeter()
                

                network.train()
                #print(f"network_index = {network_index} ,network.train()")
                self.step_count +=1

                perm = torch.randperm(self.split_whole_size)

                self.uvst_train_gpu = self.uvst_train_gpu[perm]
                self.rgb_train_gpu = self.rgb_train_gpu[perm]
                
                
                self.train_loader = [{'input': self.uvst_train_gpu[s:e], 
                                        'color': self.rgb_train_gpu[s:e]} for s, e in zip(self.start, self.end)]

                pbar = self.train_loader
                for i, data_batch in enumerate(pbar):

                    optimizer.zero_grad()
                    inputs  = data_batch["input"].cuda()
                    color = data_batch["color"].cuda()
                    #print(f"network_index = {network_index} batch_index = {i} ,data load")
                    preds_color = network(inputs.cuda())
                    #print(f"network_index = {network_index} batch_index = {i} ,infer")
                    
                    loss = 1000*torch.mean((preds_color - color) * (preds_color - color))

                    #save_dir = self.img_folder_train + f"epoch-{epoch}"

                    self.losses_batch.update(loss.item(), inputs.size(0))
                    
                
                    loss.backward()
                
                    #print(f"network_index = {network_index} batch_index = {i} ,loss.backward()")
                    optimizer.step()
                    #print(f"network_index = {network_index} batch_index = {i} ,optimizer.step()")

                scheduler.step()
                if self.grid_n < 4 :    
                    log_str = 'epoch {}/{}, network {}/{}, loss:{:4f}'.format(epoch,self.epoch_num,network_index,self.grid_n*self.grid_n-1,self.losses_batch.avg)
                    logging.info(log_str)
                elif self.grid_n >= 4 and network_index %((self.grid_n*self.grid_n)//8) == 1:
                    log_str = 'epoch {}/{}, network {}/{}, loss:{:4f}'.format(epoch,self.epoch_num,network_index,self.grid_n*self.grid_n-1,self.losses_batch.avg)
                    logging.info(log_str)
                    
                self.losses.update(self.losses_batch.avg)



            log_str = 'whole network arg : epoch {}/{}, loss:{:4f}'.format(epoch,self.epoch_num,self.losses.avg)
            logging.info(log_str)
                
            
            
            
            with torch.no_grad():

                if epoch % args.val_freq ==0 or epoch % args.test_freq ==0:
                    self.model.eval()

                if epoch % args.val_freq ==0:
                    self.val(epoch)

                # if epoch % args.save_checkpoints == 0:
                #     cpt_path = self.checkpoints + f"nelf-{epoch}.pth"
                #     torch.save(self.model.state_dict(), cpt_path)
                if epoch % args.save_checkpoints == 0:
                    # 모든 네트워크의 상태를 저장할 사전을 생성
                    state_dicts = {}
                    
                    for network_index, network in enumerate(self.model.networks):
                        # 각 네트워크의 상태 사전에 고유한 키를 할당하여 저장
                        state_dicts[f'network_{network_index}'] = network.state_dict()
                    
                    # 체크포인트 파일 경로 설정
                    cpt_path = f"{self.checkpoints}/nelf-all-networks-epoch_{epoch}.pth"
                    
                    # 모든 네트워크 상태가 포함된 사전을 파일로 저장
                    torch.save(state_dicts, cpt_path)
                    
                    print(f"모든 네트워크 모델 체크포인트가 저장되었습니다: {cpt_path}")
                # if epoch % args.test_freq ==0:
                    
                #     cam_num = np.random.randint(10)

                #     uvst_cam = self.uvst_whole[cam_num*self.w*self.h:(cam_num+1)*self.w*self.h,:]
                #     gt_colors = self.color_whole[cam_num*self.w*self.h:(cam_num+1)*self.w*self.h,:]

                #     # generate predicted camera position
                #     cam_x = np.random.uniform(self.min_x,self.max_x)
                #     cam_y = np.random.uniform(self.min_y,self.max_y)
                #     cam_z = self.uv_depth

                #     gt_img = gt_colors.reshape((self.h,self.w,3)).transpose((2,0,1))
                #     gt_img = torch.from_numpy(gt_img)

                #     # uvst_random = self.get_uvst(cam_x,cam_y,cam_z)
                #     # uvst_random_centered = self.get_uvst(cam_x,cam_y,cam_z,True)

                #     save_dir = self.img_folder_train + f"epoch-{epoch}"
                #     rm_folder_keep(save_dir)
                
                #     self.render_sample_img(self.model, uvst_cam, self.w, self.h, f"{save_dir}/recon.png")#,f"{save_dir}/recon_depth.png")
                #     # self.render_sample_img(self.model, uvst_random, self.w, self.h, f"{save_dir}/random.png")#,f"{save_dir}/random_depth.png")
                #     # self.render_sample_img(self.model, uvst_random_centered, self.w, self.h, f"{save_dir}/random_centered.png")#,f"{save_dir}/random_depth.png")
                    
                #     torchvision.utils.save_image(gt_img, f"{save_dir}/gt.png")

                #     #self.gen_gif_llff_path(f"{save_dir}/video_ani_llff.gif")
                #     #self.val(epoch)
                    
            time = Timer.stop() 
            log_time = f"time : {time} ms"
            logging.info(log_time)

        # if(args.loss_clamp):
        #     psnr_full = self.val(epoch)
        #     temp_psnr = np.asarray(psnr_full).mean()
        #     temp_epoch = epoch
        #     temp_loss = self.losses.avg
        #     self.result_tracker.add_measurement(self.index_i, self.index_j ,temp_epoch, temp_psnr,temp_loss)
        #     self.result_tracker.save()
        #     for _ in range(5):
        #         print(f" finish input epoch : i {self.index_i} , j {self.index_j} , epoch {temp_epoch} , loss {temp_loss} , psnr {temp_psnr}" )

    def val(self,epoch):
        with torch.no_grad():
            i=0
            p = []
            s = []
            l = []

            save_dir = self.img_folder_test + f"epoch-{epoch}"
            rm_folder_keep(save_dir)
            
            count = 0
            pred_imgs = []
            for i in range(self.val_size):
                grid_imgs = []
                for network_index, network in enumerate(self.model.networks):
                    
                    grid_v = network_index % self.grid_n
                    grid_h = network_index // self.grid_n
                    
                    uvst = torch.from_numpy(self.uvst_val[grid_h, grid_v,i,:,:,:]).cuda()
                    pred_color = network(uvst)
                    grid_imgs.append(pred_color)
                    
                # GPU 상의 텐서를 포함하는 리스트를 CPU로 이동시킨 후 NumPy 배열로 변환
                grid_imgs = np.array([img.cpu().numpy() for img in grid_imgs])

                pred_imgs.append(self.merge_images(grid_imgs,self.grid_n))

            # pred_imgs 리스트에 포함된 이미지들을 저장
        

# pred_imgs 리스트에 포함된 이미지들을 저장
            for count, (p_img, gt_img) in enumerate(zip(pred_imgs, self.rgb_val)):
                 # PyTorch 텐서일 경우 CPU로 이동시키고 NumPy 배열로 변환
                if isinstance(p_img, torch.Tensor):
                    p_img_np = p_img.cpu().numpy()
                else:
                    p_img_np = p_img  # 이미 NumPy 배열인 경우

                if isinstance(gt_img, torch.Tensor):
                    gt_img_np = gt_img.cpu().numpy()
                else:
                    gt_img_np = gt_img  # 이미 NumPy 배열인 경우

                # 이미지 저장 디렉토리가 없으면 생성
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                
                # 예측 이미지 저장
                p_img_pil = Image.fromarray((p_img_np * 255).astype(np.uint8))  # [0, 1] -> [0, 255] 범위 조정
                p_img_pil.save(f"{save_dir}/val_{count+1}.png")

                # 실제 이미지 저장
                gt_img_pil = Image.fromarray((gt_img_np * 255).astype(np.uint8))  # [0, 1] -> [0, 255] 범위 조정
                gt_img_pil.save(f"{save_dir}/gt_{count+1}.png")
                
                # PSNR 계산
                psnr = peak_signal_noise_ratio(gt_img_np, p_img_np, data_range=1)  # 데이터 범위 [0, 255]
                p.append(psnr)
                            

        

            

            logging.info(f'>>> val: psnr  mean {np.asarray(p).mean()} full {p}')
            # self.val_psnr.append([epoch,np.asarray(p).mean()])
            # self.val_psnr_full.append(np.array(p))

            # logging.info(f'>>> val: ssim  mean {np.asarray(s).mean()} full {s}')
            # logging.info(f'>>> val: lpips mean {np.asarray(l).mean()} full {l}')
            return p
    


    def get_ray(self,s,t):
        H = self.h
        W = self.w
        x = s
        y = t
        aspect = W/H
        u = np.linspace(-1, 1, W, dtype='float32')
        v = np.linspace(1, -1, H, dtype='float32') / aspect

        vu = list(np.meshgrid(u, v))

        u = vu[0]
        v = vu[1]
        s = np.ones_like(vu[0]) * x
        t = np.ones_like(vu[1]) * y
        uvst =np.stack((u, v, s, t), axis=-1)
        return uvst

    

    def get_render_pose(self , duration =120):
        
        n = duration
        repeats = np.linspace(0, 2*np.pi, n) 
        uvsts = []
        for r in repeats:
            uvst = self.get_ray(np.sin(r) ,np.sin(2*r))
            uvsts.append(uvst)

        return np.array(uvsts)
    
    def get_ray_6D(self,x,z,theta_y = 0):
        H = self.h
        W = self.w

        y = 0
        aspect = W/H


        # theta = np.pi
        
        # rot_z = np.array([
        #     [np.cos(theta), -np.sin(theta), 0],
        #     [np.sin(theta), np.cos(theta), 0],
        #     [0, 0, 1]
        # ])

        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        #rot = rot_z @ rot_y
        rot = rot_y
        


        u = np.linspace(-1, 1, W, dtype='float32')
        v = np.linspace(1, -1, H, dtype='float32') / aspect

        vu = list(np.meshgrid(u, v))


        u = vu[0]
        v = vu[1]
        dirs = np.stack((u, v, -np.ones_like(u)), axis=-1)
        dirs = np.sum(dirs[..., np.newaxis, :]* rot,-1)
        

        tx = np.ones_like(dirs[:,:,0:1]) *x
        ty = np.ones_like(dirs[:,:, 0:1]) *y
        tz = np.ones_like(dirs[:,:, 0:1]) *z

        vec3_xyz = np.concatenate((dirs, tx, ty, tz), axis=-1)
        
        return vec3_xyz
    

    def get_render_pose_6D(self , duration =120 , rotation = True):
        
        n = duration
        repeats = np.linspace(0, 2*np.pi, n) 
        uvsts = []
        c = 0.6063
        for r in repeats:
            if rotation:
                uvst = self.get_ray_6D(c*np.sin(r) ,c*np.sin(2*r) , r)
            else:
                uvst = self.get_ray_6D(c*np.sin(r) ,c*np.sin(2*r))
            uvsts.append(uvst)

        
   
        for r in repeats:
            if rotation:
                uvst = self.get_ray_6D(c*np.sin(2*r) ,c*np.sin(r) , r)
            else:
                uvst = self.get_ray_6D(c*np.sin(2*r) ,c*np.sin(r))
            uvsts.append(uvst)
        #breakpoint()
        return np.array(uvsts)

    def render_uvst(self, uvsts, savename):
        # uvsts를 그리드에 맞게 분할
        uvsts = self.split_images_to_grid(uvsts, self.grid_n)
        uvsts_tensor = torch.from_numpy(uvsts).cuda()

        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(savename + ".mp4", fourcc, 24.0, (self.w, self.h))

        frame_num = uvsts.shape[2]

        with torch.no_grad():
            for i in tqdm(range(frame_num), desc="Rendering Video"):
                grid_imgs = None
                for network_index, network in enumerate(self.model.networks):
                    grid_v = network_index % self.grid_n
                    grid_h = network_index // self.grid_n
                    uvst = uvsts_tensor[grid_h, grid_v, i, :, :, :].float()
                    pred_color = network(uvst)

                    if grid_imgs is None:
                        grid_imgs = pred_color.unsqueeze(0)  # 첫 번째 텐서를 기준으로 빈 텐서 생성
                    else:
                        grid_imgs = torch.cat((grid_imgs, pred_color.unsqueeze(0)), 0)  # 이후 텐서를 연결

                # 그리드 이미지를 하나의 큰 이미지로 병합
                
                view_unit = self.merge_images_tensor(grid_imgs)*255
                


                # OpenCV는 BGR 포맷을 사용하므로 색상 순서 변경
                #view_unit_bgr = cv2.cvtColor(view_unit.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                if self.use_yuv:
                    view_unit_bgr = cv2.cvtColor(view_unit.cpu().numpy().astype(np.uint8), cv2.COLOR_YCrCb2RGB)
                    
                else:
                    view_unit_bgr = cv2.cvtColor(view_unit.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                out.write(view_unit_bgr)

            out.release()

    def gen_video(self,savename):
        pass
        



    

    def render_sample_img(self,model,uvst, w, h, save_path=None,save_depth_path=None,save_flag=True):
        with torch.no_grad():
           
            uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()

            pred_color = model(uvst)
           
            pred_img = pred_color.reshape((h,w,3)).permute((2,0,1))

            if(save_flag):
                torchvision.utils.save_image(pred_img, save_path)
           
            return pred_color.reshape((h,w,3))
        
  


    def train_summaries(self):  
        self.summary_writer.add_scalar('total loss', self.losses.avg, self.step_count)

    def add_psnr_data(self):
        PsnrSaver.psnr.append(self.val_psnr)
        PsnrSaver.label.append(self.mode)

    def save_psnr_full(self):
        path = 'result/'+self.exp
        result_path = os.path.join(path, f"{self.mode}_psnr_full.npy")
        result_path_txt = os.path.join(path, f"{self.mode}_psnr_full.txt")
        np.save(result_path,np.array(self.val_psnr_full))
        np.savetxt(result_path_txt,np.array(self.val_psnr_full))

    def save_psnr_mean(self):
        path = 'result/'+self.exp
        result_path = os.path.join(path, f"{self.mode}_psnr_mean.npy")
        result_path_txt = os.path.join(path, f"{self.mode}_psnr_mean.txt")
        np.save(result_path,np.array(self.val_psnr))
        np.savetxt(result_path_txt,np.array(self.val_psnr))

    def save_psnr_data(self):
        self.add_psnr_data()
        self.save_psnr_full()
        self.save_psnr_mean()

    def load_data_grid(self,type, mode , grid_n):
        return self.split_images_to_grid(self.load_data(type,mode) , grid_n)

    def load_data(self, type, mode):
        path = os.path.join(self.data_dir, mode, type)
        
        # 디렉토리 존재 여부 확인
        if not os.path.exists(path) or not os.path.isdir(path):
            raise FileNotFoundError(f"지정된 경로를 찾을 수 없습니다: {path}")
        
        # .npy 파일 리스트 생성
        npy_files = [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith('.npy')]

        # 파일 리스트가 비어 있는지 확인
        if not npy_files:
            raise FileNotFoundError(f"{path} 디렉토리에 .npy 파일이 없습니다.")

        # 파일 이름으로 정렬
        npy_files.sort()

        # 불러온 NumPy 배열을 저장할 리스트
        arrays = []

        # 각 .npy 파일을 불러와서 리스트에 추가
        for file_path in npy_files:
            array = np.load(file_path)
            arrays.append(array)

        # 리스트에 저장된 배열들을 하나의 NumPy 배열로 결합
        combined_array = np.array(arrays)
        return combined_array

    def split_images_to_grid(self,images, grid_n):
        """
        이미지 배열을 그리드로 분할합니다.

        :param images: 이미지 배열, shape (n, h, w, 3)
        :param grid_n: 그리드의 수 (세로 및 가로 모두 동일하게 grid_n x grid_n으로 분할)
        :return: 분할된 이미지 그리드, shape (grid_n, grid_n, n, h/grid_n, w/grid_n, 3)
        """
        n, h, w, c = images.shape
        grid_h = h // grid_n  # 그리드 한 칸의 높이
        grid_w = w // grid_n  # 그리드 한 칸의 너비

        # (grid_n, grid_n, n, grid_h, grid_w, c) 형태로 이미지 재구성
        grid_images = np.zeros((grid_n, grid_n, n, grid_h, grid_w, c), dtype=images.dtype)
        
        for i in range(grid_n):
            for j in range(grid_n):
                # 이미지를 그리드에 맞게 분할하여 할당
                grid_images[i, j] = images[:, i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w, :]

        return grid_images
    
    def load_check_points_multi_select(self,epoch):
        ckpt_path = os.path.join(self.checkpoints,f"nelf-all-networks-epoch_{epoch}.pth")
        self.iter = epoch

        print(f"Load weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path)

                # 각 네트워크별로 상태를 로드합니다.
        for network_index, network in enumerate(self.model.networks):
            try:
                        # 체크포인트 파일에서 해당 네트워크의 상태 사전을 가져와 로드합니다.
                network_state_dict = ckpt[f'network_{network_index}']
                network.load_state_dict(network_state_dict)
            except KeyError as e:
                print(f"Error loading state_dict for network {network_index}: {e}")


    def load_check_points_multi(self):
        ckpt_paths = glob.glob(self.checkpoints + "*.pth")
        self.iter = 0
        latest_ckpt_path = None
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                print(ckpt_path)
                # 파일명에서 숫자를 추출하기 위해 정규 표현식 사용
                match = re.search(r'epoch_(\d+).pth', ckpt_path)
                if match:
                    ckpt_id = int(match.group(1))
                    if ckpt_id > self.iter:
                        self.iter = ckpt_id
                        latest_ckpt_path = ckpt_path

            if latest_ckpt_path:
                print(f"Load weights from {latest_ckpt_path}")
                ckpt = torch.load(latest_ckpt_path)

                # 각 네트워크별로 상태를 로드합니다.
                for network_index, network in enumerate(self.model.networks):
                    try:
                        # 체크포인트 파일에서 해당 네트워크의 상태 사전을 가져와 로드합니다.
                        network_state_dict = ckpt[f'network_{network_index}']
                        network.load_state_dict(network_state_dict)
                    except KeyError as e:
                        print(f"Error loading state_dict for network {network_index}: {e}")

    def load_check_points(self):
        ckpt_paths = glob.glob(self.checkpoints+"*.pth")
        self.iter=0
        if len(ckpt_paths) > 0:
            for ckpt_path in ckpt_paths:
                print(ckpt_path)
                ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                self.iter = max(self.iter, ckpt_id)
            ckpt_name = f"./{self.checkpoints}/nelf-{self.iter}.pth"
        # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
        print(f"Load weights from {ckpt_name}")
        
        ckpt = torch.load(ckpt_name)
        try:
            self.model.load_state_dict(ckpt)
        except:
            tmp = DataParallel(self.model)
            tmp.load_state_dict(ckpt)
            self.model.load_state_dict(tmp.module.state_dict())
            del tmp
            
    def merge_images_tensor(self, grids):
        n = self.grid_n  # 그리드의 한 축에 있는 이미지의 수

        # grids 텐서의 모양에서 이미지의 높이, 너비, 채널 수를 추출합니다.
        _, height, width, channels = grids.shape

        # 최종 이미지의 높이와 너비를 계산합니다.
        final_height = n * height
        final_width = n * width

        # 최종 이미지를 저장할 빈 텐서를 생성합니다.
        final_image = torch.zeros((final_height, final_width, channels), device=grids.device)

        # 각 이미지를 올바른 위치에 복사합니다.
        for i in range(n):
            for j in range(n):
                # 현재 이미지의 인덱스를 계산합니다.
                idx = i * n + j
                # 해당 이미지를 최종 이미지 텐서에 복사합니다.
                final_image[i*height:(i+1)*height, j*width:(j+1)*width, :] = grids[idx]

        return final_image
    

    def merge_images(self,grids, n):
        """
        n x n 그리드 이미지를 하나의 큰 이미지로 병합합니다.
        :param grids: n x n 그리드 이미지 리스트
        :param n: 그리드의 수 (세로 및 가로)
        :return: 병합된 이미지
        """
        # 각 행을 생성하기 위해 n개의 이미지를 수평으로 연결합니다.
        rows = [np.concatenate(grids[i*n:(i+1)*n], axis=1) for i in range(n)]
        # 모든 행을 수직으로 연결하여 전체 이미지를 만듭니다.
        merged_image = np.concatenate(rows, axis=0)
        return merged_image




if __name__ == '__main__':

    

    args = parser.parse_args()


    

    uvst = train(args ,"uvst")
    uvst.train_step(args)

#     lp = LineProfiler()

#     # 프로파일링할 함수 실행 (여기서는 메서드에 필요한 모든 인자를 전달)
#     lp.runctx('uvst.train_step(args)', globals(), locals())

#     # 프로파일링 결과 출력
#     lp.print_stats()


# if __name__ == "__main__":
#     # ArgumentParser 설정과 args 파싱
   
#     args = parser.parse_args()

#     # train 클래스 인스턴스화 및 초기화
#     uvst = train(args, "uvst")

#     cProfile을 사용한 프로파일링
#     cProfile.runctx("uvst.train_step(args)", globals(), locals(), "profile.prof")

#     # 프로파일링 결과를 pstats 객체로 로딩
#     stats = pstats.Stats("profile.prof")

#     # 프로파일링 결과 출력 또는 다른 형식으로 저장
#     stats.strip_dirs().sort_stats("cumtime").print_stats()

#     # gprof2dot으로 시각화를 위해 .prof 파일을 .dot 파일로 변환
#     os.system("gprof2dot -f pstats profile.prof | dot -Tpng -o profile.png")

#     # 생성된 시각화 이미지 파일 확인
#     print("프로파일링 결과 이미지 파일 'profile.png'이 생성되었습니다.")
