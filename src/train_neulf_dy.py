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
import time
import math

sys.path.insert(0, os.path.abspath('./'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
import argparse
import cv2
from torch.utils.tensorboard import SummaryWriter

from src.model_dy import Nerf4D_relu_ps
from src.model_addon2 import Nerf4D_relu_ps_addon2
from src.model_addon2_2 import Nerf4D_relu_ps_addon2_2
from src.model_addon4 import Nerf4D_relu_ps_addon4

import torchvision
import glob
import torch.optim as optim
from utils import rm_folder,AverageMeter,rm_folder_keep

import imageio
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
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

def get_rotaion_matirx(theta):

    rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
    ])
    return rotation_matrix.T


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp name')
parser.add_argument('--data_dir',type=str, 
                    default = 'dataset/Ollie/',help='data folder name')
parser.add_argument('--batch_size',type=int, default = 8192,help='normalize input')
parser.add_argument('--test_freq',type=int,default=1,help='test frequency')
parser.add_argument('--save_checkpoints',type=int,default=1,help='checkpoint frequency')
parser.add_argument('--whole_epoch',type=int,default=120,help='checkpoint frequency')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument("--factor", type=int, default=1, help='downsample factor for LLFF images')
parser.add_argument('--img_scale',type=int, default= 1, help= "devide the image by factor of image scale")
# parser.add_argument('--norm_fac',type=float, default=1, help= "normalize the data uvst")
# parser.add_argument('--st_norm_fac',type=float, default=1, help= "normalize the data uvst")
parser.add_argument('--work_num',type=int, default= 15, help= "normalize the data uvst")
parser.add_argument('--lr_pluser',type=int, default = 100,help = 'scale for dir')
parser.add_argument('--lr',type=float,default=5e-04,help='learning rate')
parser.add_argument('--loadcheckpoints', action='store_true', default = False)
parser.add_argument('--st_depth',type=int, default= 0, help= "st depth")
parser.add_argument('--uv_depth',type=int, default= 0.0, help= "uv depth")
parser.add_argument('--rep',type=int, default=1)
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--mlp_width', type=int, default = 256)
#imlab

parser.add_argument('--renderpose_only', action='store_true' ,default = False)
parser.add_argument('--timecheck', action='store_true' ,default = False)
parser.add_argument('--test_render', action='store_true' ,default = False)

parser.add_argument('--load_exp', type=str, default = None,help = 'load exp name')
parser.add_argument('--addon', type=int, default= 0, help= "addon mode")
parser.add_argument('--loadckptnum',type=int, default = 0,help = 'epoch number to load')
parser.add_argument('--val', action='store_true', default = False)

class train():
    def __init__(self,args):
        print("1102 333")
        process = psutil.Process(os.getpid())
         # set gpu id
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))
        print('start : ',process.memory_info().rss / (1024 ** 2), "MB")

        # data root
        self.data_root = args.data_dir
        # data_img = os.path.join(args.data_dir,'images_{}'.format(args.factor)) 

        # if args.addon == 2: 
        #     self.model = Nerf4D_relu_ps_addon2(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        # elif args.addon == 22: 
        #     self.model = Nerf4D_relu_ps_addon2_2(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        # elif args.addon == 4: 
        #     self.model = Nerf4D_relu_ps_addon4(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        # else: self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width)
        for name, module in self.model.named_children():
                print(name, module)

        self.exp = 'Exp_'+args.exp_name

        # tensorboard writer
        self.summary_writer = SummaryWriter("src/tensorboard/"+self.exp)

        # save img
        self.save_img = True
        self.img_folder_train = 'result/'+self.exp+'/train/'
        self.img_folder_test = 'result/'+self.exp+'/test/'
        self.checkpoints = 'result/'+self.exp+'/checkpoints/'
        
        # make folder
        rm_folder_keep(self.img_folder_train)
        rm_folder_keep(self.img_folder_test)
        rm_folder_keep(self.checkpoints)

        handlers = [logging.StreamHandler()]
        dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
        handlers.append(logging.FileHandler('result/'+self.exp+f'/{dt_string}.log', mode='w'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-5s %(message)s',
            datefmt='%m-%d %H:%M:%S', handlers=handlers,
        )

        # load checkpoints
        if args.load_exp is not None:
            # load_ckpt
            self.loadexp = 'Exp_'+args.load_exp
            self.loadexp_folder = 'result/'+self.loadexp
            self.loadckpt = self.loadexp_folder+'/checkpoints/'
        # load log and lr
            self.load_exp_ckpt(args.loadckptnum)
        
        # load checkpoints
        if(args.loadcheckpoints):
            self.load_check_points()
        
        self.model = self.model.cuda()

        # height and width
        # image_paths = glob.glob(f"{data_img}/*.png")
        # sample_img = cv2.imread(image_paths[0])
        # self.h = int(sample_img.shape[0]/args.img_scale)
        # self.w = int(sample_img.shape[1]/args.img_scale)
        self.h = 720
        self.w = 720
        # self.img_scale = args.img_scale

        self.step_count = 0

        # # load nelf data
        # print(f"Start loading...")
        # split='train'
        # # center
        # self.uvst_whole   = np.load(f"{data_root}/uvst{split}.npy")
        # print("Stop loading...")
        # self.uvst_whole_len = self.uvst_whole.shape[0]
        # self.uvst_whole_gpu    = torch.tensor(self.uvst_whole).float()
        # self.uvst_whole = None
        # self.color_whole   = np.load(f"{data_root}/rgb{split}.npy")
        # self.color_whole_gpu      = torch.tensor(self.color_whole).float()
        # self.color_whole = None
        # self.color_whole = None

        # self.start, self.end = [], []
        # s = 0
        # while s < self.uvst_whole_len:
        #     self.start.append(s)
        #     s += args.batch_size
        #     self.end.append(min(s, self.uvst_whole_len))
       
        # split='val'
        # self.uvst_whole_val   = np.load(f"{data_root}/uvst{split}.npy")
        # self.color_whole_val   = np.load(f"{data_root}/rgb{split}.npy")
        
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.vis_step = 1
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995) 
        self.epoch_num = args.whole_epoch
        # mpips eval
        self.lpips = lpips.LPIPS(net='vgg')
        
        frames_folders = []
        for item in sorted(os.listdir(self.data_root)):
            if item.startswith('frames_') and os.path.isdir(os.path.join(self.data_root, item)):
                frames_folders.append(item)
        self.framescount = len(frames_folders)
        if not frames_folders:
            print("No folders starting with 'frames_' found.")
        print("frames count:", self.framescount)
        
    def train_step(self,args):
        frame_list = list(range(1, self.framescount + 1))
        for epoch in range(0,self.epoch_num+1):
            if(args.loadcheckpoints):
                epoch += self.iter
            random.shuffle(frame_list)
            print(frame_list)
            f = 0
            num = 30
            for i in range(0, len(frame_list), num):
                sublist = frame_list[i:i+num]
                print(sublist)
                self.all_uvst_whole =[]
                self.all_color_whole = []
                
                for frame in sublist:
                    print(f"Start loading frame {frame}...")
                    split='train'
                    # center
                    self.uvst_whole   = np.load(f"{self.data_root}/{frame:04d}_uvst{split}.npy")
                    self.all_uvst_whole.append(self.uvst_whole)

                    self.color_whole   = np.load(f"{self.data_root}/{frame:04d}_rgb{split}.npy")
                    self.all_color_whole.append(self.color_whole)

                self.all_uvst_whole = np.concatenate(self.all_uvst_whole, axis=0)
                self.all_color_whole = np.concatenate(self.all_color_whole, axis=0)
                self.uvst_whole_gpu    = torch.tensor(np.array(self.all_uvst_whole)).float()
                self.uvst_whole_len = self.all_uvst_whole.shape[0]
                self.all_uvst_whole = None
                print(f"uvst loaded...")
                
                self.color_whole_gpu      = torch.tensor(np.array(self.all_color_whole)).float()
                self.color_whole = None
                print(f"rgb loaded...")
                
                print('uvst length:', self.uvst_whole_len)
                self.start, self.end = [], []
                s = 0
                while s < self.uvst_whole_len:
                    self.start.append(s)
                    s += args.batch_size
                    self.end.append(min(s, self.uvst_whole_len))
            
                self.losses = AverageMeter()
                self.losses_rgb = AverageMeter()
                self.losses_rgb_super = AverageMeter()
                self.losses_depth = AverageMeter()

                self.model.train()
                self.step_count +=1
                f += len(sublist)
                print('start training...')
                perm = torch.randperm(self.uvst_whole_len)
                self.uvst_whole_gpu = self.uvst_whole_gpu[perm]
                self.color_whole_gpu = self.color_whole_gpu[perm]
                
                self.train_loader = [{'input': self.uvst_whole_gpu[s:e], 
                                        'color': self.color_whole_gpu[s:e]} for s, e in zip(self.start, self.end)]

                pbar = tqdm(self.train_loader)

                for i, data_batch in enumerate(pbar):

                    self.optimizer.zero_grad()
                    inputs  = data_batch["input"].cuda()
                    color = data_batch["color"].cuda()

                    preds_color = self.model(inputs.cuda())
                    
                    loss_rgb = 1000*torch.mean((preds_color - color) * (preds_color - color))
                    loss = loss_rgb 

                    self.losses.update(loss.item(), inputs.size(0))
                    self.losses_rgb.update(loss_rgb.item(),inputs.size(0))
                
                    loss.backward()
                
                    self.optimizer.step()
                    log_str = 'epoch {}/{}, frame:{}/{}, {}/{}, lr:{}, loss:{:4f}'.format(
                        epoch,self.epoch_num,f,self.framescount,i+1,len(self.start),
                        self.optimizer.param_groups[0]['lr'],self.losses.avg)
                    # if (i+1) % 2000 == 0:
                    #     logging.info(log_str)
                logging.info(log_str)
                self.scheduler.step()
                
                with torch.no_grad():
                    self.model.eval()
                    if epoch % args.test_freq ==0:
                        self.save_dir = self.img_folder_test + f"epoch-{epoch}"
                        rm_folder_keep(self.save_dir)
                        self.val(epoch)
                    
                    if epoch % args.save_checkpoints == 0:
                        cpt_path = self.checkpoints + f"nelf-{epoch}.pth"
                        torch.save(self.model.state_dict(), cpt_path)
                        print(f"Saved weights to {cpt_path}")
                    
 
    def train_summaries(self):
        self.summary_writer.add_scalar('total loss', self.losses.avg, self.step_count)
        
    def get_uvst2(self,cam_x,cam_y,cam_z,rotmaty, rotation_matric = None ,center_flag=False ):

        theta = np.pi
        theta_y = rotmaty
        rot_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        rot = rot_z @ rot_y
        W = self.w
        H = self.h

        uvst_tmp = []
        aspect = W/H
       
        u = np.linspace(-1, 1, W, dtype='float32')
        v = np.linspace(1, -1, H, dtype='float32') / aspect
        vu = list(np.meshgrid(u, v))

        u = vu[0]
        v = vu[1] 
        dirs = np.stack((u, v, -np.ones_like(u)), axis=-1)
        dirs = np.sum(dirs[..., np.newaxis, :]* rot,-1)
        dirs = np.array(dirs)
        dirs = np.reshape(dirs,(-1,3))
        
        x = np.ones_like(vu[0]) * cam_x
        y = np.ones_like(vu[0]) * cam_y
        z = np.ones_like(vu[0]) * cam_z
        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        x = np.reshape(x,(-1,1))
        y = np.reshape(y,(-1,1))
        z = np.reshape(z,(-1,1))

        concatenated_array = np.concatenate((dirs, x, y, z), axis=1)
        uvst_tmp.append(concatenated_array)

        #uvst_tmp = np.concatenate([uvst_tmp, concatenated_array])
        uvst_tmp = concatenated_array

        uvst_tmp = np.asarray(uvst_tmp)
        uvst_tmp = np.reshape(uvst_tmp,(-1,6))
        data_uvst = uvst_tmp

        return data_uvst                

    def val(self,epoch):
        with torch.no_grad():
            val_frames = [frame for frame in range(self.framescount) if os.path.exists(os.path.join(self.data_root, f"{frame:04d}_uvstval.npy"))]
            uvst_val_list = []
            color_val_list = []
            for val_frame in val_frames:
                uvst_val = np.load(f"{self.data_root}/{val_frame:04d}_uvstval.npy")
                color_val = np.load(f"{self.data_root}/{val_frame:04d}_rgbval.npy")
                uvst_val_list.append(uvst_val)
                color_val_list.append(color_val)
            self.uvst_whole_val = np.concatenate(uvst_val_list, axis=0)
            self.color_whole_val = np.concatenate(color_val_list, axis=0)
            print('val loaded')
            
            i=0
            p = []
            s = []
            l = []

            count = 0
            while i < self.uvst_whole_val.shape[0]:
                end = i+self.w*self.h
                uvst = self.uvst_whole_val[i:end]
                uvst = torch.from_numpy(uvst.astype(np.float32)).cuda()

                pred_color = self.model(uvst)
                gt_color = self.color_whole_val[i:end]

                # write to file
                pred_img = pred_color.reshape((self.h,self.w,3)).permute((2,0,1))
                gt_img   = torch.tensor(gt_color).reshape((self.h,self.w,3)).permute((2,0,1))
                

                torchvision.utils.save_image(pred_img,f"{self.save_dir}/test_{count}.png")
                torchvision.utils.save_image(gt_img,f"{self.save_dir}/gt_{count}.png")

                pred_color = pred_color.cpu().numpy()
                
                psnr = peak_signal_noise_ratio(gt_color, pred_color, data_range=1)
                ssim = structural_similarity(gt_color.reshape((self.h,self.w,3)), pred_color.reshape((self.h,self.w,3)), data_range=pred_color.max() - pred_color.min(),multichannel=True)
                lsp  = self.lpips(pred_img.cpu(),gt_img) 

                p.append(psnr)
                s.append(ssim)
                l.append(lsp.numpy().item())

                i = end
                count+=1

            logging.info(f'>>>  val: psnr  mean {np.asarray(p).mean()} full {p}')
            logging.info(f'>>>  val: ssim  mean {np.asarray(s).mean()} full {s}')
            logging.info(f'>>>  val: lpips mean {np.asarray(l).mean()} full {l}')
            self.psnr_file_path = 'result/' + self.exp + '/psnr_values.txt'
            with open(self.psnr_file_path, 'a') as psnr_file:
                psnr_file.write(f'{epoch} {np.asarray(p).mean()} {p}\n')
                
            return p

    def train_summaries(self):
        self.summary_writer.add_scalar('total loss', self.losses.avg, self.step_count)

     
    def load_exp_ckpt(self, loadckptnum):
        if loadckptnum == 0:
            ckpt_paths = glob.glob(self.loadckpt+"*.pth")
            self.iter=0
            if len(ckpt_paths) > 0:
                for ckpt_path in ckpt_paths:
                    print(ckpt_path)
                    ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                    self.iter = max(self.iter, ckpt_id)
                ckpt_name = f"./{self.loadckpt}/nelf-{self.iter}.pth"
            # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
        else:
            self.iter = loadckptnum
            ckpt_name = f"./{self.loadckpt}/nelf-{loadckptnum}.pth"
            
        print(f"Load weights from {ckpt_name}") 
        ckpt = torch.load(ckpt_name)
        
        try:
            self.model.load_state_dict(ckpt, strict=False)
        except:
            tmp = DataParallel(self.model)
            tmp.load_state_dict(ckpt)
            self.model.load_state_dict(tmp.module.state_dict())
            # self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            del tmp
        
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

if __name__ == '__main__':

    args = parser.parse_args()
    m_train = train(args)
    if args.val is True:
        m_train.val('test')
        sys.exit()
    m_train.train_step(args)
