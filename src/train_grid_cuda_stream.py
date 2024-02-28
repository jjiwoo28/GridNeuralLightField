
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
parser.add_argument('--st_depth',type=int, default= 0, help= "st depth")
parser.add_argument('--uv_depth',type=int, default= 0.0, help= "uv depth")
parser.add_argument('--rep',type=int, default=1)
parser.add_argument('--mlp_depth', type=int, default = 4)
parser.add_argument('--mlp_width', type=int, default = 64)
parser.add_argument('--load_epoch', type=int, default = 0)

parser.add_argument('--grid_n', type=int, default = 2)
parser.add_argument('--loss_clamp', action='store_true', default = False)
parser.add_argument('--loss_clamp_value', type=int , default=5)





class grid_cuda_stream(train):

     def train_step(self,args):


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
        self.split_whole_size = np.reshape(self.uvst_train[0, 0,:,:,:,:],(-1,4)).shape[0]
        if self.split_whole_size > self.batch_size:
            self.batch_size = self.split_whole_size
        while s < self.split_whole_size:
            self.start.append(s)
            s += args.batch_size
            self.end.append(min(s, self.split_whole_size))
       
        # optimizer
       
        self.vis_step = 1
        
        Timer.start()
        self.epoch_num = args.whole_epoch
        start_epoch = 0
        if(args.loadcheckpoints):
                    start_epoch = self.iter
        streams = [torch.cuda.Stream() for _ in range(len(self.model.networks))]
        stream_step = 100

        for epoch in range(start_epoch, start_epoch+self.epoch_num+1 , stream_step):
            self.losses = AverageMeter()

            for network_index, (network, optimizer, scheduler) in enumerate(zip(self.model.networks, self.model.optimizers, self.model.schedulers)):
                with torch.cuda.stream(streams[network_index]):
                    for e in range(epoch,epoch+stream_step):
                        # 여기서부터 스트림 내의 연산
                        grid_v = network_index % self.grid_n
                        grid_h = network_index // self.grid_n

                        self.uvst_train_gpu = torch.tensor(np.reshape(self.uvst_train[grid_h, grid_v,:,:,:,:],(-1,4))).float().cuda(non_blocking=True)
                        self.rgb_train_gpu = torch.tensor(np.reshape(self.rgb_train[grid_h, grid_v,:,:,:,:],(-1,3))).float().cuda(non_blocking=True)

                        self.losses_batch = AverageMeter()

                        network.train()
                        self.step_count += 1

                        perm = torch.randperm(self.split_whole_size).cuda()

                        self.uvst_train_gpu = self.uvst_train_gpu[perm]
                        self.rgb_train_gpu = self.rgb_train_gpu[perm]

                        self.train_loader = [{'input': self.uvst_train_gpu[s:e], 'color': self.rgb_train_gpu[s:e]} for s, e in zip(self.start, self.end)]

                        for i, data_batch in enumerate(self.train_loader):
                            optimizer.zero_grad()
                            inputs = data_batch["input"].cuda(non_blocking=True)
                            color = data_batch["color"].cuda(non_blocking=True)

                            preds_color = network(inputs)

                            loss = 1000 * torch.mean((preds_color - color) * (preds_color - color))
                            self.losses_batch.update(loss.item(), inputs.size(0))

                            loss.backward()
                            optimizer.step()

                        scheduler.step()
                        log_str = 'epoch {}/{}, network {}/{}, loss:{:4f}'.format(e, self.epoch_num, network_index, self.grid_n*self.grid_n-1, self.losses_batch.avg)
                        logging.info(log_str)

                        # 로그 출력은 스트림의 연산이 완료된 후에 수행
                        # if self.grid_n < 4:    
                        #     log_str = 'epoch {}/{}, network {}/{}, loss:{:4f}'.format(e, self.epoch_num, network_index, self.grid_n*self.grid_n-1, self.losses_batch.avg)
                        #     logging.info(log_str)
                        #     #torch.cuda.current_stream().synchronize()
                        # elif self.grid_n >= 4 and network_index % ((self.grid_n*self.grid_n)//8) == 1:
                        #     #torch.cuda.current_stream().synchronize()
                        #     log_str = 'epoch {}/{}, network {}/{}, loss:{:4f}'.format(e, self.epoch_num, network_index, self.grid_n*self.grid_n-1, self.losses_batch.avg)
                        #     logging.info(log_str)
                            
                        self.losses.update(self.losses_batch.avg)

                # 각 스트림별 작업 완료를 기다립니다.
                streams[network_index].synchronize()

            log_str = 'whole network arg : epoch {}/{}, loss:{:4f}'.format(epoch, self.epoch_num, self.losses.avg)
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
                
            time = Timer.stop() 
            log_time = f"time : {time} ms"
            logging.info(log_time)

if __name__ =="__main__":
    
    args = parser.parse_args()  

    uvst = grid_cuda_stream(args ,"uvst")
    uvst.train_step(args)
