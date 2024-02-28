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


from torch.utils.tensorboard import SummaryWriter

from src.model import Nerf4D_relu_ps
import torchvision
import glob
import torch.optim as optim
from utils import eval_uvst,rm_folder,AverageMeter,rm_folder_keep,eval_trans
from src.tracker import ResultTracker 

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
parser.add_argument('--whole_epoch',type=int,default=31,help='checkpoint frequency')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument("--factor", type=int, default=4, help='downsample factor for LLFF images')
parser.add_argument('--img_scale',type=int, default= 1, help= "devide the image by factor of image scale")
parser.add_argument('--norm_fac',type=float, default=1, help= "normalize the data uvst")
parser.add_argument('--st_norm_fac',type=float, default=1, help= "normalize the data uvst")
parser.add_argument('--work_num',type=int, default= 15, help= "normalize the data uvst")
parser.add_argument('--lr_pluser',type=int, default = 100,help = 'scale for dir')
parser.add_argument('--lr',type=float,default=5e-04,help='learning rate')
parser.add_argument('--loadcheckpoints', action='store_true', default = False)
parser.add_argument('--st_depth',type=int, default= 0, help= "st depth")
parser.add_argument('--uv_depth',type=int, default= 0.0, help= "uv depth")
parser.add_argument('--rep',type=int, default=1)
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--mlp_width', type=int, default = 128)
parser.add_argument('--index_i', type=int, default = 0)
parser.add_argument('--index_j', type=int, default = 0)
parser.add_argument('--grid_n', type=int, default = 2)
parser.add_argument('--loss_clamp', action='store_true', default = False)
parser.add_argument('--loss_clamp_value', type=int , default=5)


class PsnrSaver:
    # 클래스의 정적 속성 정의
    psnr = []
    label = []
    result_path = None


def exp_name_parser(text):
    """
    주어진 문자열에서 'i[j]_j[k]' 패턴 이전까지의 문자열을 추출합니다.
    
    :param text: 입력 문자열
    :return: 추출된 문자열 또는 패턴이 없으면 None
    """
    match = re.search(r'(.*)_i\d+_j\d+', text)
    if match:
        return match.group(1)
    else:
        return None


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
        print("240107 fixed 666666666666666666")
        self.mode = mode
        # data root
        data_root = args.data_dir
        n = args.grid_n
        
        self.grig_n = args.grid_n 

        self.index_i = args.index_i
        self.index_j = args.index_j
        self.exp_name = args.exp_name

        self.result_json_path = f"result_json/{exp_name_parser(self.exp_name)}.json"
        self.result_tracker = ResultTracker(args.grid_n ,self.result_json_path)
        self.result_tracker.load()

        self.loss_clamp_value  = args.loss_clamp_value
        
        data_img = os.path.join(args.data_dir,'images_{}'.format(args.factor)) 

        if(mode == "uvst"):
            self.model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width,input_dim=4)
        
        # uv and st depth
        self.uv_depth = args.uv_depth
    
        # normalize factor/home/icte_gpu_1/NeuLF_Super/result/Exp_uvsv_1_test_clampLoss_knight32_lr_d4w32_i15_j15

        self.norm_fac = args.norm_fac
        self.st_norm_fac = args.st_norm_fac
        self.test_freq = args.test_freq
        self.val_freq = args.val_freq

        PsnrSaver.result_path = os.path.join("result",args.exp_name + "PsnrResult")
        if(mode == "uvst"):
            self.exp = 'Exp_uvsv_'+args.exp_name
        elif (mode == "uvsthv"):
            self.exp = 'Exp_uvsthv_'+args.exp_name

        # tensorboard writer
        self.summary_writer = SummaryWriter("src/tensorboard/"+self.exp)

        # save img
        self.save_img = True
        self.img_folder_train = 'result/'+self.exp+'/train/'
        self.img_folder_test = 'result/'+self.exp+'/test/'
        self.checkpoints = 'result/'+self.exp+'/checkpoints/'

        self.val_psnr = []
        self.val_psnr_full = []

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
        if(args.loadcheckpoints):
            self.load_check_points()

        self.model = self.model.cuda()

        # height and width
        image_paths = glob.glob(f"{data_img}/*.png")
        sample_img = cv2.imread(image_paths[0])
        self.h = int((sample_img.shape[0]/args.img_scale)/n)
        self.w = int((sample_img.shape[1]/args.img_scale)/n)

        self.img_scale = args.img_scale

        self.step_count = 0

        # load nelf data
        print(f"Start loading...")
        split='train'
        # center

        if(mode == "uvst"):
            self.uvst_whole   = np.load(f"{data_root}/uvst{args.grid_n}_{args.index_i}_{args.index_j}_{split}.npy") / args.norm_fac
        elif (mode == "uvsthv"):
            self.uvst_whole   = np.load(f"{data_root}/uvsthv{args.grid_n}_{args.index_i}_{args.index_j}_{split}.npy") / args.norm_fac


        for _ in range(20):
            print(self.uvst_whole.shape)

        self.uvst_whole[:,2:] /= self.st_norm_fac

        # norm to 0 to 1
        self.uvst_min = self.uvst_whole.min()
        self.uvst_max = self.uvst_whole.max()
        #self.uvst_whole = (self.uvst_whole - self.uvst_min) / (self.uvst_max - self.uvst_min) * 2 - 1.0
        
        # center color
        self.color_whole   = np.load(f"{data_root}/rgb{args.grid_n}_{args.index_i}_{args.index_j}_{split}.npy")
       
        self.trans        = np.load(f"{data_root}/trans{split}.npy")
        self.intrinsic    = np.load(f"{data_root}/k{split}.npy")
        self.fdepth       = np.load(f"{data_root}/fdepth{split}.npy") # center object
        self.render_pose  = np.load(f"{data_root}/Render_pose{split}.npy")#render path spiral
        
        self.st_depth     = -self.fdepth
   
        self.uvst_whole = np.concatenate([self.uvst_whole]*args.rep, axis=0)
        self.color_whole = np.concatenate([self.color_whole]*args.rep, axis=0)

        split='val'

        if(mode == "uvst"):
            self.uvst_whole_val   = np.load(f"{data_root}/uvst{args.grid_n}_{args.index_i}_{args.index_j}_{split}.npy") / args.norm_fac
        elif (mode == "uvsthv"):
            self.uvst_whole_val   = np.load(f"{data_root}/uvsthv{args.grid_n}_{args.index_i}_{args.index_j}_{split}.npy") / args.norm_fac
        self.uvst_whole_val[:,2:] /= self.st_norm_fac
        self.color_whole_val   = np.load(f"{data_root}/rgb{args.grid_n}_{args.index_i}_{args.index_j}_{split}.npy")

        print("print(self.uvst_whole_val.shape)")
        for _ in range(20):
            print(self.uvst_whole_val.shape)
        #self.uvst_whole_val = (self.uvst_whole_val - self.uvst_min) / (self.uvst_max - self.uvst_min) * 2 - 1.0
        
        self.trans_val        = np.load(f"{data_root}/trans{split}.npy")
        self.intrinsic_val    = np.load(f"{data_root}/k{split}.npy")
        self.fdepth_val       = np.load(f"{data_root}/fdepth{split}.npy") # center object
        self.render_pose_val  = np.load(f"{data_root}/Render_pose{split}.npy")#render path spiral
        self.st_depth_val     = -self.fdepth
        print("Stop loading...")

        rays_whole = np.concatenate([self.uvst_whole, self.color_whole], axis=1)
       
        self.min_u,self.max_u,self.min_v,self.max_v,self.min_s,self.max_s,self.min_t,self.max_t = eval_uvst(rays_whole)
        self.max_x,self.min_x,self.max_y,self.min_y = eval_trans(self.trans)

        logging.info(f"u[{self.min_u},{self.max_u}], v[{self.min_v}, {self.max_v}], s[{self.min_s}, {self.max_s}], t[{self.min_t}, {self.max_t}]")

        # data loader
        self.uvst_whole_gpu    = torch.tensor(self.uvst_whole).float()
      
        self.color_whole_gpu      = torch.tensor(self.color_whole).float()
     
        self.start, self.end = [], []
        s = 0
        while s < self.uvst_whole.shape[0]:
            self.start.append(s)
            s += args.batch_size
            self.end.append(min(s, self.uvst_whole.shape[0]))
       
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))
       
        self.vis_step = 1
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999) 

        self.epoch_num = args.whole_epoch
    
        # mpips eval
        #self.lpips = lpips.LPIPS(net='vgg') 

    def train_step(self,args):
        epoch_tmp = None
        for epoch in range(0,self.epoch_num+1):

            temp_loss =None
            temp_psnr =None
            temp_epoch =None
            
            if(args.loadcheckpoints):
                epoch += self.iter

            epoch_tmp = epoch
            self.losses = AverageMeter()
            self.losses_rgb = AverageMeter()
            self.losses_rgb_super = AverageMeter()
            self.losses_depth = AverageMeter()

            self.model.train()
            self.step_count +=1

            perm = torch.randperm(self.uvst_whole.shape[0])
            self.uvst_whole_gpu = self.uvst_whole_gpu[perm]
            self.color_whole_gpu = self.color_whole_gpu[perm]
            
            
            self.train_loader = [{'input': self.uvst_whole_gpu[s:e], 
                                    'color': self.color_whole_gpu[s:e]} for s, e in zip(self.start, self.end)]
            save_dir = self.img_folder_train + f"epoch-{epoch}"

            pbar = self.train_loader
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
                log_str = 'epoch {}/{}, {}/{}, lr:{}, loss:{:4f}'.format(
                    epoch,self.epoch_num,i+1,len(self.start),
                    self.optimizer.param_groups[0]['lr'],self.losses.avg,)
           
                if (i+1) % 2000 == 0:
                    logging.info(log_str)

            if(self.losses.avg < self.loss_clamp_value and args.loss_clamp and epoch>=100 ):
                psnr_full = self.val(epoch)
                temp_psnr = np.asarray(psnr_full).mean()
                temp_epoch = epoch
                temp_loss = self.losses.avg
                self.result_tracker.add_measurement(self.index_i, self.index_j ,temp_epoch, temp_psnr,temp_loss)
                self.result_tracker.save(self.result_json_path)
                for _ in range(5):
                    print(f"clamp loss : i {self.index_i} , j {self.index_j} , epoch {temp_epoch} , loss {temp_loss}, psnr {temp_psnr}" )
                return True

                
                
            logging.info(log_str)
            self.scheduler.step()
            
            with torch.no_grad():
                self.model.eval()

                if epoch % args.val_freq ==0:
                    self.val(epoch)
                if epoch % args.test_freq ==0:
                    
                    cam_num = np.random.randint(10)

                    uvst_cam = self.uvst_whole[cam_num*self.w*self.h:(cam_num+1)*self.w*self.h,:]
                    gt_colors = self.color_whole[cam_num*self.w*self.h:(cam_num+1)*self.w*self.h,:]

                    # generate predicted camera position
                    cam_x = np.random.uniform(self.min_x,self.max_x)
                    cam_y = np.random.uniform(self.min_y,self.max_y)
                    cam_z = self.uv_depth

                    gt_img = gt_colors.reshape((self.h,self.w,3)).transpose((2,0,1))
                    gt_img = torch.from_numpy(gt_img)

                    # uvst_random = self.get_uvst(cam_x,cam_y,cam_z)
                    # uvst_random_centered = self.get_uvst(cam_x,cam_y,cam_z,True)

                    save_dir = self.img_folder_train + f"epoch-{epoch}"
                    rm_folder_keep(save_dir)
                
                    self.render_sample_img(self.model, uvst_cam, self.w, self.h, f"{save_dir}/recon.png")#,f"{save_dir}/recon_depth.png")
                    # self.render_sample_img(self.model, uvst_random, self.w, self.h, f"{save_dir}/random.png")#,f"{save_dir}/random_depth.png")
                    # self.render_sample_img(self.model, uvst_random_centered, self.w, self.h, f"{save_dir}/random_centered.png")#,f"{save_dir}/random_depth.png")
                    
                    torchvision.utils.save_image(gt_img, f"{save_dir}/gt.png")

                    #self.gen_gif_llff_path(f"{save_dir}/video_ani_llff.gif")
                    #self.val(epoch)

                if epoch % args.save_checkpoints == 0:
                    cpt_path = self.checkpoints + f"nelf-{epoch}.pth"
                    torch.save(self.model.state_dict(), cpt_path)
        if(args.loss_clamp):
            psnr_full = self.val(epoch)
            temp_psnr = np.asarray(psnr_full).mean()
            temp_epoch = epoch
            temp_loss = self.losses.avg
            self.result_tracker.add_measurement(self.index_i, self.index_j ,temp_epoch, temp_psnr,temp_loss)
            self.result_tracker.save(self.result_json_path)
            for _ in range(5):
                print(f" finish input epoch : i {self.index_i} , j {self.index_j} , epoch {temp_epoch} , loss {temp_loss} , psnr {temp_psnr}" )
        

    def get_uvst(self,cam_x,cam_y,cam_z, center_flag=False):

        t = np.asarray([cam_x,cam_y,cam_z])

        center = np.array([0,0,-self.fdepth])

        # get rotation matrix
        if(center_flag):
            cur_p = normalize(center-t)
            cur_o = np.array([0,0,-1])
            rot_mat = rotation_matrix_from_vectors(cur_o,cur_p)
        else:
            rot_mat = np.eye(3,dtype=float)

        c2w = np.concatenate((rot_mat,np.expand_dims(t,1)),1)

        ray_o,ray_d = get_rays(self.h,self.w,self.intrinsic,c2w)

        ray_o = np.reshape(ray_o,(-1,3))
        ray_d = np.reshape(ray_d,(-1,3))

        plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)

        # interset radius plane
        p_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),np.shape(ray_o))
        p_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),np.shape(ray_o))

        # interset radius plane 
        inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)

        inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)
       
        data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)
        
        data_uvst /= self.norm_fac

        data_uvst = (data_uvst - self.uvst_min)/(self.uvst_max-self.uvst_min) * 2 -1.0

        return data_uvst           

    def val(self,epoch):
        with torch.no_grad():
            i=0
            p = []
            s = []
            l = []

            save_dir = self.img_folder_test + f"epoch-{epoch}"
            rm_folder_keep(save_dir)
            
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
                
                
                torchvision.utils.save_image(pred_img,f"{save_dir}/test_{count}.png")
                torchvision.utils.save_image(gt_img,f"{save_dir}/gt_{count}.png")

                pred_color = pred_color.cpu().numpy()
                
                psnr = peak_signal_noise_ratio(gt_color, pred_color, data_range=1)
                # ssim = structural_similarity(gt_color.reshape((self.h,self.w,3)), pred_color.reshape((self.h,self.w,3)), data_range=pred_color.max() - pred_color.min(),multichannel=True)
                # lsp  = self.lpips(pred_img.cpu(),gt_img) 

                p.append(psnr)
                # s.append(ssim)
                # l.append(lsp.numpy().item())

                i = end
                count+=1

            logging.info(f'>>> val: psnr  mean {np.asarray(p).mean()} full {p}')
            self.val_psnr.append([epoch,np.asarray(p).mean()])
            self.val_psnr_full.append(np.array(p))

            # logging.info(f'>>> val: ssim  mean {np.asarray(s).mean()} full {s}')
            # logging.info(f'>>> val: lpips mean {np.asarray(l).mean()} full {l}')
            return p

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

    def gen_gif_llff_path(self,savename):
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
    
        view_group = []

        for i, c2w in enumerate(tqdm(self.render_pose)):
            ray_o, ray_d = get_rays_np(self.h, self.w, self.intrinsic, c2w)

            ray_o = np.reshape(ray_o,(-1,3))
            ray_d = np.reshape(ray_d,(-1,3))

            plane_normal = np.broadcast_to(np.array([0.0,0.0,1.0]),ray_o.shape)

            p_uv = np.broadcast_to(np.array([0.0,0.0,self.uv_depth]),np.shape(ray_o))
            p_st = np.broadcast_to(np.array([0.0,0.0,self.st_depth]),np.shape(ray_o))

            # interset radius plane 
            inter_uv = rayPlaneInter(plane_normal,p_uv,ray_o,ray_d)
            inter_st = rayPlaneInter(plane_normal,p_st,ray_o,ray_d)

            data_uvst = np.concatenate((inter_uv[:,:2],inter_st[:,:2]),1)
            data_uvst /= self.norm_fac
            data_uvst[:,2:] /= self.st_norm_fac

            data_uvst = (data_uvst - self.uvst_min)/(self.uvst_max - self.uvst_min) * 2 -1.0
            
            view_unit = self.render_sample_img(self.model,data_uvst,self.w,self.h,None,None,False)

            view_unit *= 255

            view_unit       = view_unit.cpu().numpy().astype(np.uint8)

            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))

            view_unit       = imageio.core.util.Array(view_unit)
            view_group.append(view_unit)

        imageio.mimsave(savename, view_group,fps=30)
        out.release()

   
    

if __name__ == '__main__':

    

    args = parser.parse_args()


    

    uvst = train(args ,"uvst")
    uvst.train_step(args)
   

