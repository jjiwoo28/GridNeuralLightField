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

import torch
import numpy as np
import argparse
import os
import glob
import sys
import time
sys.path.insert(0, os.path.abspath('./'))
from src.model_dy import Nerf4D_relu_ps
from utils import rm_folder, rm_folder_keep
import torchvision
import logging
import cv2
import imageio
from tqdm import tqdm

parser = argparse.ArgumentParser() # museum,column2
parser.add_argument('--exp_name',type=str, default = 'Ollie_d8_w256',help = 'exp_name')
parser.add_argument('--load_exps', nargs='+', type=str, default=['Ollie_d8_w256'], help='load_exps')
parser.add_argument('--gpuid',type=str, default = '0',help='data folder name')
parser.add_argument('--mlp_depth', type=int, default = 8)
parser.add_argument('--mlp_width', type=int, default = 256)
parser.add_argument('--scale', type=int, default = 4)
parser.add_argument('--img_form',type=str, default = '.png',help = 'exp name')
parser.add_argument('--addon', type=int, default= 0, help= "addon mode")

class demo_rgb():
    def __init__(self,args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
        print('>>> Using GPU: {}'.format(args.gpuid))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)-5s %(message)s')
        self.load_exps = []
        self.checkpoints = []
        self.models = []
        for i, load_exp in enumerate(args.load_exps):
            # data_root
            self.load_exps.append('Exp_' + load_exp)
            self.checkpoints.append('result/Exp_' + load_exp + '/checkpoints/')
        
        self.load_check_points()
        for i in range(len(args.load_exps)):
            self.models[i] = self.models[i].cuda()
            
        self.w = 720
        self.h = 720
        
        self.folder_path = 'demo_rgb_result'
        rm_folder_keep(self.folder_path)
        print(self.folder_path)
        # self.gen_zline_vid_llff_path(f"{self.folder_path}/line_video_ani_llff_z1", 0.5, 0, 3, 90, 180, 0)
        # self.gen_zline_vid_llff_path(f"{self.folder_path}/line_video_ani_llff_z2", 1.5, 0, 3, 90, 180, 0)
        # self.gen_zline_vid_llff_path(f"{self.folder_path}/line_video_ani_llff_z3", 2.5, 0, 3, 90, 180, 0)
        # self.gen_zline_vid_llff_path(f"{self.folder_path}/line_video_ani_llff_z4", 1, 0, 3, 90, 180, 0)
        # self.gen_rot_vid_llff_path(f"{self.folder_path}/rot_video_ani_llff_360", 1.5, 1.5, 0, 360, 120, 0)
        # self.gen_rot_gif_llff_path(f"{self.folder_path}/rot_video_ani_llff_row", 0, 0, 0, 90)
        # self.gen_360_vid_llff_path(f"{self.folder_path}/{args.exp_name}", 0)
        # self.gen_my_vid_llff_path(f"{self.folder_path}/{args.exp_name}")
        
        self.gen_dy_cen_fixed_vid_path(f"{self.folder_path}/{args.exp_name}_cf")
        # self.gen_dy_unknowntime_vid_path(f"{self.folder_path}/{args.exp_name}_ut")
        # self.gen_dy_cen_rot_vid_path(f"{self.folder_path}/{args.exp_name}_cr")
        # self.gen_dy_vid_path(f"{self.folder_path}/{args.exp_name}_d")
        # self.gen_cen_blended_imgs_path(f"{self.folder_path}/{args.exp_name}")
        # self.gen_cen_rot_blended_vid_llff_path(f"{self.folder_path}/cen_rot_blended_video_ani_llff")
        
    
        
    def load_check_points(self):
        for i in range(len(args.load_exps)):
            ckpt_paths = glob.glob(self.checkpoints[i]+"*.pth")
            self.iter=0 
            if len(ckpt_paths) > 0:
                for ckpt_path in ckpt_paths:
                    print(ckpt_path)
                    ckpt_id = int(os.path.basename(ckpt_path).split(".")[0].split("-")[1])
                    self.iter = max(self.iter, ckpt_id)
                ckpt_name = f"./{self.checkpoints[i]}/nelf-{self.iter}.pth"
            # ckpt_name = f"{self.checkpoints}nelf-{self.fourier_epoch}.pth"
            print(f"Load weights from {ckpt_name}")
            ckpt = torch.load(ckpt_name)
            # if args.addon == 2: 
            #     model = Nerf4D_relu_ps_addon2(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width).cuda()
            # elif args.addon == 22: 
            #     model = Nerf4D_relu_ps_addon2_2(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width).cuda()
            # elif args.addon == 4: 
            #     model = Nerf4D_relu_ps_addon4(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width).cuda()
            # else: model = Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width).cuda()
            model= Nerf4D_relu_ps(D=args.mlp_depth,W=args.mlp_width,input_ch=args.mlp_width).cuda()
            model.load_state_dict(ckpt)
            self.models.append(model)
    
    def gen_dy_unknowntime_vid_path(self, savename):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 30.0, (self.w,self.h))
        frames = np.linspace(-1, 18, 360)
        angle = np.linspace(170, 190, 360)
        for f, a in tqdm(zip(frames, angle), total=len(frames)):
            view_unit = self.make_view_unit(0, 0, 0, a, f, 0)
            out.write(cv2.cvtColor(view_unit, cv2.COLOR_RGB2BGR))
        out.release()
        print("dy_cen_ut_vid_finish")
        
    def gen_dy_cen_rot_vid_path(self, savename):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 30.0, (self.w,self.h))
        frames = np.linspace(1, 16, 225)
        angle = np.linspace(195, 165, 225)
        for f, a in tqdm(zip(frames, angle), total=len(frames)):
            view_unit = self.make_view_unit(0, 0, 0, a, f, 0)
            out.write(cv2.cvtColor(view_unit, cv2.COLOR_RGB2BGR))
        out.release()
        print("dy_cen_sp_vid_finish")
        
    def gen_dy_cen_fixed_vid_path(self, savename):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 30.0, (self.w,self.h))
        frames = np.linspace(1, 87, 860)
        for f in tqdm(zip(frames), total=len(frames)):
            view_unit = self.make_view_unit(0, 0, 0, 180, f, 0)
            out.write(cv2.cvtColor(view_unit, cv2.COLOR_RGB2BGR))
        out.release()
        print("dy_cen_fixed_vid_finish")
        
    def gen_dy_vid_path(self, savename):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 30.0, (self.w,self.h))
        frames = np.linspace(1, 87, 860)
        xs = np.linspace(-3.5, 3.5, 860)
        zs = np.linspace(-0.7, 0.7, 860)
        angle = np.linspace(195, 165, 860)
        for f, x, z, a in tqdm(zip(frames, xs, zs, angle), total=len(frames)):
            view_unit = self.make_view_unit(x, 0, z, a, f, 0)
            out.write(cv2.cvtColor(view_unit, cv2.COLOR_RGB2BGR))
        out.release()
        print("dy_vid_finish")
        
    def get_uvst2(self,cam_x,cam_y,cam_z,rotmaty,frame):

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
        # aspect = W/H
       
        u = np.linspace(-1, 1, W, dtype='float32')
        v = np.linspace(1, -1, H, dtype='float32')
        # v = np.linspace(1, -1, H, dtype='float32') / aspect
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
        frame_array = np.full_like(x, frame)
        frame_array = np.array(frame_array)
        frame_array = np.reshape(frame_array,(-1,1))
        
        concatenated_array = np.concatenate((dirs, x, y, z, frame_array), axis=1)
        uvst_tmp.append(concatenated_array)
        #uvst_tmp = np.concatenate([uvst_tmp, concatenated_array])
        # uvst_tmp = concatenated_array
        uvst_tmp = np.asarray(uvst_tmp)
        uvst_tmp = np.reshape(uvst_tmp,(-1,7))
        data_uvst = uvst_tmp

        return data_uvst
    
    def make_view_unit(self, cam_x, cam_y, cam_z, rot_y, frame, model_idx):
        data_uvst = self.get_uvst2(cam_x, cam_y, cam_z, np.deg2rad(rot_y), frame)
        uvst = torch.from_numpy(data_uvst.astype(np.float32)).cuda()
        with torch.no_grad():
            pred_color = self.models[model_idx](uvst)
        view_unit = pred_color.reshape((self.h, self.w, 3))
        view_unit *= 255
        view_unit = view_unit.cpu().numpy().astype(np.uint8)
        return view_unit
    
    def make_blended_view_unit(self, cam_x, cam_y, cam_z, rot_srt, rot_end, rot_y, frame, model_idx_1, model_idx_2):
        data_uvst = self.get_uvst2(cam_x, cam_y, cam_z, np.deg2rad(rot_srt + rot_y), frame)
        uvst = torch.from_numpy(data_uvst.astype(np.float32)).cuda()
        rot_angle = rot_end - rot_srt
        with torch.no_grad():
            if rot_y < rot_angle / 3:
                pred_color = self.models[model_idx_1](uvst)
            elif rot_y < 2 * rot_angle / 3:
                pred_color1 = self.models[model_idx_1](uvst)
                pred_color2 = self.models[model_idx_2](uvst)
                alpha = (rot_y - rot_angle / 3) / (rot_angle / 3)
                pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
            else:
                pred_color = self.models[model_idx_2](uvst)
        view_unit = pred_color.reshape((self.h, self.w, 3))
        view_unit *= 255
        view_unit = view_unit.cpu().numpy().astype(np.uint8)
        return view_unit
    
    def gen_xline_vid_llff_path(self, savename, x_srt, x_end, rot_y, frame, model_idx):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        cam_x = np.linspace(x_srt, x_end, frame)
        logging.info(cam_x)
        for x in cam_x:
            view_unit = self.make_view_unit(x, 0, 0, rot_y, model_idx)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        out.release()
        print("gen_xline_vid_finish")
    
    def gen_zline_vid_llff_path(self, savename, x, z_srt, z_end, rot_y, frame, model_idx):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        cam_z = np.linspace(z_srt, z_end, frame)
        logging.info(cam_z)
        for z in cam_z:
            view_unit = self.make_view_unit(x, 0, z, rot_y, model_idx)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        out.release()
        print("gen_zline_vid_finish")
    
    def gen_cen_imgs_path(self, savename):
        rm_folder_keep(savename)
        cam_rot = np.linspace(0, 360, 720)
        logging.info(cam_rot)
        for idx, r in enumerate(cam_rot):
            view_unit = self.make_view_unit(1.5, 0, 1.5, r, 0)
            view_unit_rgb = cv2.cvtColor(view_unit, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(savename, f"frame_{idx:04d}.png"), view_unit_rgb)
        print("gen_cen_imgs_finish")
        
    def gen_cen_blended_imgs_path(self, savename):
        rm_folder_keep(savename)
        cam_rot = np.linspace(0, 360, 360)
        logging.info(cam_rot)
        
        for idx, r in enumerate(cam_rot):
            data_uvst = self.get_uvst2(0, 0, 0, np.deg2rad(r))
            uvst = torch.from_numpy(data_uvst.astype(np.float32)).cuda()
            with torch.no_grad():
                if r < 30:
                    pred_color = self.models[0](uvst)
                elif r < 60:
                    pred_color1 = self.models[0](uvst)
                    pred_color2 = self.models[1](uvst)
                    alpha = (r - 30) / 30
                    pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
                elif r < 120:
                    pred_color = self.models[1](uvst)
                elif r < 150:
                    pred_color1 = self.models[1](uvst)
                    pred_color2 = self.models[2](uvst)
                    alpha = (r - 120) / 30
                    pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
                elif r < 210:
                    pred_color = self.models[2](uvst)
                elif r < 240:
                    pred_color1 = self.models[2](uvst)
                    pred_color2 = self.models[3](uvst)
                    alpha = (r - 210) / 30
                    pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
                elif r < 300:
                    pred_color = self.models[3](uvst)
                elif r < 330:
                    pred_color1 = self.models[3](uvst)
                    pred_color2 = self.models[0](uvst)
                    alpha = (r - 300) / 30
                    pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
                else:
                    pred_color = self.models[0](uvst)  
            view_unit = pred_color.reshape((self.h, self.w, 3))
            view_unit *= 255
            view_unit = view_unit.cpu().numpy().astype(np.uint8)
            view_unit_rgb = cv2.cvtColor(view_unit, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(savename, f"frame_{idx:04d}.png"), view_unit_rgb)
        print("gen_cen_imgs_finish")
        
    def gen_rot_vid_llff_path(self, savename, cam_x, cam_z, rot_srt, rot_end, frame, model_idx):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        cam_rot = np.linspace(rot_srt, rot_end, frame)
        logging.info(cam_rot)
        for r in cam_rot:
            view_unit = self.make_view_unit(cam_x, 0, cam_z, r, model_idx)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        out.release()
        print("gen_rot_vid_finish")
        
    def gen_90deg_blended_vid_llff_path(self, savename, cam_x, cam_z, rot_srt, frame, model_idx_1, model_idx_2):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        rot_end = rot_srt+90
        cam_rot = np.linspace(rot_srt, rot_end, frame)
        logging.info(cam_rot)
        for r in cam_rot:
            view_unit = self.make_blended_view_unit(cam_x, 0, cam_z, rot_srt, rot_end, r, model_idx_1, model_idx_2)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        out.release()
        print("finish")
        
    def gen_cen_rot_blended_vid_llff_path(self, savename):
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        cam_rot = np.linspace(0, 360, 360)
        logging.info(cam_rot)
        
        for r in cam_rot:
            data_uvst = self.get_uvst2(1.5, 0, 1.5, np.deg2rad(r))
            uvst = torch.from_numpy(data_uvst.astype(np.float32)).cuda()
            with torch.no_grad():
                if r < 30:
                    pred_color = self.models[0](uvst)
                elif r < 60:
                    pred_color1 = self.models[0](uvst)
                    pred_color2 = self.models[1](uvst)
                    alpha = (r - 30) / 30
                    pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
                elif r < 120:
                    pred_color = self.models[1](uvst)
                elif r < 150:
                    pred_color1 = self.models[1](uvst)
                    pred_color2 = self.models[2](uvst)
                    alpha = (r - 120) / 30
                    pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
                elif r < 210:
                    pred_color = self.models[2](uvst)
                elif r < 240:
                    pred_color1 = self.models[2](uvst)
                    pred_color2 = self.models[3](uvst)
                    alpha = (r - 210) / 30
                    pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
                elif r < 300:
                    pred_color = self.models[3](uvst)
                elif r < 330:
                    pred_color1 = self.models[3](uvst)
                    pred_color2 = self.models[0](uvst)
                    alpha = (r - 300) / 30
                    pred_color = (1-alpha)*pred_color1 + alpha*pred_color2
                else:
                    pred_color = self.models[0](uvst)  
            view_unit = pred_color.reshape((self.h, self.w, 3))
            view_unit *= 255
            view_unit = view_unit.cpu().numpy().astype(np.uint8)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        out.release()
        print("finish")
        
    def gen_360_vid_llff_path(self, savename, model_idx):
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        cam_rot = np.linspace(0, 90, 90)
        cam_inc = np.linspace(0, 3, 90)
        cam_dec = cam_inc[::-1]
        logging.info(cam_rot)
        logging.info(cam_inc)
        logging.info(cam_dec)
        
        for x in cam_dec:
            view_unit = self.make_view_unit(x, 0, 0, 0, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        
        for r in cam_rot:
            view_unit = self.make_view_unit(0, 0, 0, r, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        
        for z in cam_inc:
            view_unit = self.make_view_unit(0, 0, z, 90, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        
        for r in cam_rot:
            view_unit = self.make_view_unit(0, 0, 3, r + 90, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
                
        for x in cam_inc:
            view_unit = self.make_view_unit(x, 0, 3, 180, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
        for r in cam_rot: 
            view_unit = self.make_view_unit(3, 0, 3, r + 180, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
        for z in cam_dec:
            view_unit = self.make_view_unit(3, 0, z, 270, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
        for r in cam_rot:
            view_unit = self.make_view_unit(3, 0, 0, r + 270, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
        out.release()
        print("finish")
       
    def gen_my_vid_llff_path(self, savename):
        
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out   = cv2.VideoWriter(savename+".mp4", fourcc, 24.0, (self.w,self.h))
        cam_rot = np.linspace(0, 90, 90)
        cam_inc = np.linspace(0, 3, 90)
        cam_dec = cam_inc[::-1]
        logging.info(cam_rot)
        logging.info(cam_inc)
        logging.info(cam_dec)
        
        for x in cam_dec:
            view_unit = self.make_view_unit(x, 0, 0, 0, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        
        for r in cam_rot:
            view_unit = self.make_blended_view_unit(0, 0, 0, 0, 90, r, 0, 1)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        
        for z in cam_inc:
            view_unit = self.make_view_unit(0, 0, z, 90, 1)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
        
        for r in cam_rot:
            view_unit = self.make_blended_view_unit(0, 0, 3, 90, 180, r, 1, 2)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
                
        for x in cam_inc:
            view_unit = self.make_view_unit(x, 0, 3, 180, 2)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
        for r in cam_rot: 
            view_unit = self.make_blended_view_unit(3, 0, 3, 180, 270, r, 2, 3)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
        for z in cam_dec:
            view_unit = self.make_view_unit(3, 0, z, 270, 3)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
        for r in cam_rot:
            view_unit = self.make_blended_view_unit(3, 0, 0, 270, 360, r, 3, 0)
            out.write(cv2.cvtColor(view_unit,cv2.COLOR_RGB2BGR))
            
        out.release()
        print("finish")

        
if __name__ == '__main__':

    args = parser.parse_args()

    unit = demo_rgb(args)

    print("finish")
 
