
from train_neulf_grid import train
from train_neulf_grid import AverageMeter
from mytimer import Timer
import argparse


import sys
import os
import logging
from datetime import datetime

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

parser.add_argument('--render_only', action='store_true', default = False)
parser.add_argument('--use_yuv', action='store_true', default = False)

parser.add_argument('--use_6D', action='store_true', default = False)
parser.add_argument('--st_depth',type=int, default= 0, help= "st depth")
parser.add_argument('--uv_depth',type=int, default= 0.0, help= "uv depth")
parser.add_argument('--rep',type=int, default=1)
parser.add_argument('--mlp_depth', type=int, default = 4)
parser.add_argument('--mlp_width', type=int, default = 64)
parser.add_argument('--load_epoch', type=int, default = 0)

parser.add_argument('--grid_n', type=int, default = 2)
parser.add_argument('--padding', type=int, default = 8)
parser.add_argument('--loss_clamp', action='store_true', default = False)
parser.add_argument('--loss_clamp_value', type=int , default=5)






class nlf_grid_padding(train):

    def train_step(self,args):
        self.padding = args.padding

        self.exp = f"{self.exp}_padding{self.padding}" 

        if self.use_6D:
            self.uvst_train = self.load_data_grid_padding("xyzxyz" , "train",self.grid_n)
            self.uvst_val = self.load_data_grid("xyzxyz" , "val",self.grid_n)
            print("load xyzxyz")
        else:
            self.uvst_train = self.load_data_grid_padding("uvst" , "train",self.grid_n)
            self.uvst_val = self.load_data_grid("uvst" , "val",self.grid_n)


        # if args.use_yuv:
        #     self.rgb_train = self.load_data_grid_padding("yuv" , "train",self.grid_n)
        #     self.rgb_val = self.load_data("yuv" , "val")
        # else:
        #     self.rgb_train = self.load_data_grid_padding("rgb" , "train",self.grid_n)
        #     self.rgb_val = self.load_data("rgb" , "val")

        self.rgb_train = self.load_data_grid_padding("rgb" , "train",self.grid_n)
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
            print(f"epoch = {epoch}")
            self.losses = AverageMeter()
            for network_index, (network, optimizer , scheduler) in enumerate(zip(self.model.networks, self.model.optimizers , self.model.schedulers)):
                print(f"network_index = {network_index}")
                grid_v = network_index % self.grid_n
                grid_h = network_index // self.grid_n

                self.uvst_train_gpu = torch.tensor(np.reshape(self.uvst_train[grid_h, grid_v,:,:,:,:],(-1,self.input_dim))).float()
                self.rgb_train_gpu = torch.tensor(np.reshape(self.rgb_train[grid_h, grid_v,:,:,:,:],(-1,3))).float()

                

                 
                self.losses_batch = AverageMeter()
                

                network.train()
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

                    preds_color = network(inputs.cuda())
                    
                    loss = 1000*torch.mean((preds_color - color) * (preds_color - color))

                    #save_dir = self.img_folder_train + f"epoch-{epoch}"

                    self.losses_batch.update(loss.item(), inputs.size(0))
                    
                
                    loss.backward()
                
                    optimizer.step()

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
                
            time = Timer.stop() 
            log_time = f"time : {time} ms"
            logging.info(log_time)


   
    def load_data_grid_padding(self,type, mode , grid_n):
        return self.split_images_to_grid_with_padding(self.load_data(type,mode) , grid_n)
    

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
    



    
    def split_images_to_grid_with_padding(self, images, grid_n, pad=8):
        pad = self.padding
        n, h, w, c = images.shape
        grid_h = h // grid_n  # 그리드 한 칸의 높이
        grid_w = w // grid_n  # 그리드 한 칸의 너비

        # 패딩을 고려한 새로운 그리드 크기 계산
        padded_grid_h = grid_h + 2*pad
        padded_grid_w = grid_w + 2*pad

        # (grid_n, grid_n, n, padded_grid_h, padded_grid_w, c) 형태로 이미지 재구성
        grid_images = np.zeros((grid_n, grid_n, n, padded_grid_h, padded_grid_w, c), dtype=images.dtype)


        test_path = 'test'
        if os.path.exists(test_path):
            print("path already exists")
        else:
            os.makedirs(test_path)
    
        for i in range(grid_n):
            for j in range(grid_n):
                # 이미지 분할 부분 계산
                # start_h_padding = i*grid_h - pad
                # end_h_padding = (i+1)*grid_h + pad
                # start_w_padding = j*grid_w - pad
                # end_w_padding = (j+1)*grid_w + pad
                start_h = max(i*grid_h - pad, 0)
                end_h = min((i+1)*grid_h + pad, h)
                start_w = max(j*grid_w - pad, 0)
                end_w = min((j+1)*grid_w + pad, w)

                # 분할된 이미지를 그리드에 할당
                # 패딩이 필요한 경우, 자동으로 0으로 채워짐
                segment = images[:, start_h:end_h, start_w:end_w, :]
                
                # if(i==0 and j==7):
                #     breakpoint()

                if(j == 0):
                    zero_padding = np.zeros((segment.shape[0],segment.shape[1],pad,segment.shape[-1]))
                    segment = np.concatenate((zero_padding,segment), axis=2)
                if(j == grid_n-1):
                    zero_padding = np.zeros((segment.shape[0],segment.shape[1],pad,segment.shape[-1]))
                    segment = np.concatenate((segment,zero_padding), axis=2)
                if(i == 0):
                    zero_padding = np.zeros((segment.shape[0],pad,segment.shape[2],segment.shape[-1]))
                    segment = np.concatenate((zero_padding,segment), axis=1)
                if(i == grid_n-1):
                    zero_padding = np.zeros((segment.shape[0],pad,segment.shape[2],segment.shape[-1]))
                    segment = np.concatenate((segment,zero_padding), axis=1)

                grid_images[i, j, :, :, :, :] = segment

                # save_path =os.path.join(test_path,f"{i}{j}.png")
                # cv2.imwrite(save_path, segment[0,:,:,:]*255)

                


        
        return grid_images

if __name__ =="__main__":
    
    args = parser.parse_args()  

    uvst = nlf_grid_padding(args ,"uvst")
    uvst.train_step(args)
