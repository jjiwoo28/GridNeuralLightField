
from train_neulf_grid import train
from train_neulf_grid import AverageMeter
from mytimer import Timer
import argparse

from utils import rm_folder_keep
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import glob
from torch.utils.tensorboard import SummaryWriter
from multi_each_yuv import YUVNetworks

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


parser.add_argument('--padding', type=int, default = 8)
parser.add_argument('--loss_clamp', action='store_true', default = False)
parser.add_argument('--loss_clamp_value', type=int , default=5)






class nlf_grid_each_yuv(train):
    def __init__(self,args , mode = "uvst"):
         # set gpu id
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))
        print("240205_888")
    
        self.mode = mode
        # data root
        
        self.data_dir = args.data_dir
       
        
        self.batch_size = args.batch_size
        self.use_yuv = args.use_yuv

        data_img = os.path.join(args.data_dir,'images_{}'.format(args.factor)) 
 
        self.exp = f"{args.exp_name}_yuv_d{args.mlp_depth}w{args.mlp_width}" 

        #json process
        # self.result_json_path = f"result_json/grid_n-{args.grid_n},d-{args.mlp_depth},w-{args.mlp_width}.json"
        # self.result_tracker = ResultTracker(args.grid_n ,self.result_json_path)
        # self.result_tracker.load()

        # self.loss_clamp_value  = args.loss_clamp_value
        
        # PsnrSaver.result_path = os.path.join("result",args.exp_name + "PsnrResult")

        #model setting

        self.model = YUVNetworks( D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width,input_dim=4)
        

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
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
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
            self.gen_video(save_dir)
            sys.exit()

    def train_step(self,args):


        self.uvst_train = self.load_data("uvst" , "train")
        self.uvst_val = self.load_data("uvst" , "val")

        if args.use_yuv:
            self.rgb_train = self.load_data("yuv" , "train")
            self.rgb_val = self.load_data("yuv" , "val")
        else:
            self.rgb_train = self.load_data("yuv" , "train")
            self.rgb_val = self.load_data("yuv" , "val")
            # self.rgb_train = self.load_data("rgb" , "train")
            # self.rgb_val = self.load_data("rgb" , "val")
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
        self.split_whole_size = np.reshape(self.uvst_train,(-1,4)).shape[0]
        
        if self.split_whole_size < self.batch_size:
            self.batch_size = self.split_whole_size
        while s < self.split_whole_size:
            self.start.append(s)
            s += args.batch_size
            self.end.append(min(s, self.split_whole_size))
       
        # optimizer
       
        self.vis_step = 1
        breakpoint()
       
        self.epoch_num = args.whole_epoch
        start_epoch = 0
        if(args.loadcheckpoints):
                    start_epoch = self.iter

        Timer.start()

        self.uvst_train_gpu_whole = torch.tensor(np.reshape(self.uvst_train,(-1,4))).float()
        self.rgb_train_gpu_whole = torch.tensor(np.reshape(self.rgb_train,(-1,3))).float()

        for epoch in range(start_epoch,start_epoch+self.epoch_num+1):
            self.losses = AverageMeter()
            perm = torch.randperm(self.split_whole_size)
            for network_index, (network, optimizer , scheduler) in enumerate(zip(self.model.networks, self.model.optimizers , self.model.schedulers)):
                
                self.losses_batch = AverageMeter()

                network.train()
                self.step_count +=1

                

                self.uvst_train_gpu = self.uvst_train_gpu_whole[perm]
                self.rgb_train_gpu = self.rgb_train_gpu_whole[perm , network_index]
                # breakpoint()
                
                self.train_loader = [{'input': self.uvst_train_gpu[s:e], 
                                        'color': self.rgb_train_gpu[s:e]} for s, e in zip(self.start, self.end)]

                pbar = self.train_loader
                for i, data_batch in enumerate(pbar):

                    optimizer.zero_grad()
                    inputs  = data_batch["input"].cuda()
                    color = data_batch["color"].cuda()

                    preds_color = network(inputs.cuda())
                    
                    #
                    loss = 1000*torch.mean((preds_color - color) * (preds_color - color))
                    #save_dir = self.img_folder_train + f"epoch-{epoch}"
                    if i % 100==0:
                        print(preds_color[100,:])

                    self.losses_batch.update(loss.item(), inputs.size(0))
                    
                
                    loss.backward()
                
                    optimizer.step()
                    log_str = 'epoch {}/{}, {}/{}, lr:{}, loss:{:4f}'.format(
                    epoch,self.epoch_num,i+1,len(self.start),
                    optimizer.param_groups[0]['lr'],self.losses_batch.avg,)
                    if (i+1) % 2000 == 0:
                        logging.info(log_str)

                scheduler.step()
            
                log_str = 'epoch {}/{}, network {}/{}, loss:{:4f}'.format(epoch,self.epoch_num,network_index,3,self.losses_batch.avg)
                logging.info(log_str)
                
                self.losses.update(self.losses_batch.avg)



            log_str = 'whole network arg : epoch {}/{}, loss:{:4f}'.format(epoch,self.epoch_num,self.losses.avg)
            logging.info(log_str)
                
            
            
            
            with torch.no_grad():

                if epoch % args.val_freq ==0 or epoch % args.test_freq ==0:
                    self.model.eval()

                if epoch % args.val_freq ==0 and epoch != 0:
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
                    color_tensor = None    

                    for sigle_incdex , single in enumerate(network.single_models):
                        
                        pred_color = single(uvst)
                        if sigle_incdex ==0:
                            color_tensor = pred_color
                        else:
                            color_tensor = torch.cat((color_tensor ,pred_color),dim=2)
                            

                    grid_imgs.append(color_tensor)
                    
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
                


        
    

if __name__ =="__main__":
    
    args = parser.parse_args()  

    uvst = nlf_grid_each_yuv(args ,"uvst")
    uvst.train_step(args)
