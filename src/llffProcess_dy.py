import numpy as np
import os, argparse, imageio  
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str, default = 'dataset/Ollie/',help = 'exp name')
# parser.add_argument("--factor", type=int, default=1, help='downsample factor for LLFF images')

if __name__ == "__main__":
    print("llffprocess_dy")
    args = parser.parse_args()
    data_dir  = args.data_dir
        
    frames_folders = []
    for item in sorted(os.listdir(data_dir)):
        if item.startswith('frames_') and os.path.isdir(os.path.join(data_dir, item)):
            frames_folders.append(item)
    framescount = len(frames_folders)
    if not frames_folders:
        print("No folders starting with 'frames_' found.")
    print("frames count:", framescount)
    
    ## load poses_bounds
    poses_arr = np.load(os.path.join(data_dir, 'poses_bounds.npy'))
    if not os.path.exists(os.path.join(data_dir, 'poses_bounds.npy')):
        print( poses_arr, 'does not exist.' )
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    
    ## load frames
    imgfiles = []
    for folder in frames_folders:
        folder_path = os.path.join(data_dir, folder)
        images = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        imgfiles.extend(images)
    print("imgs count:", len(imgfiles))
    print("imgs per frame:", len(imgfiles)//framescount)
    if poses.shape[-1] != len(imgfiles) // framescount:
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles)//framescount, poses.shape[-1]) )
    
    def imread_with_progress(f):
        imgs = []
        for f in tqdm(f, desc="Reading images"):
            if f.endswith('png'):
                imgs.append(imageio.imread(f, ignoregamma=True)[...,:3]/255.)
            else:
                imgs.append(imageio.imread(f)[...,:3]/255.)
        return np.stack(imgs, -1)
    
    # sh = (720, 720)
    # poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    poses = poses.astype(np.float32)
    
    val_frames = [8, 16, 24]
    val_images = [11, 25, 41]
    
    val_idx = np.asarray([k for k in range(poses.shape[0])if (k+1) in val_images])
    train_idx = np.asarray([k for k in range(poses.shape[0])if (k+1) not in val_images])
    all_idx = np.asarray([k for k in range(poses.shape[0])])
    
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    # Cast intrinsics to correct types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    
    def save_idx(idx, label, frame):
        uvst_path = f"{data_dir}{frame:04d}_uvst.npy"
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
            frame_array = np.full_like(x, frame)
            frame_array = np.array(frame_array)
            frame_array = np.reshape(frame_array,(-1,1))
            concatenated_array = np.concatenate((dirs, x, y, z, frame_array), axis=1)
            uvst_tmp.append(concatenated_array)

        uvst_tmp = np.asarray(uvst_tmp)
        uvst_tmp = np.reshape(uvst_tmp,(-1,7))
        print("uvst = ", uvst_tmp.shape)
        np.save(uvst_path.replace('.npy', f'{label}.npy'), uvst_tmp)
        print(f'{uvst_path} saved')
        
    def save_idx_rgb(idx, label, frame):
        rgb_path = f"{data_dir}{frame:04d}_rgb.npy"
        rgb = np.reshape(images[idx], (-1, 3))
        print("rgb = ", rgb.shape)
        np.save(rgb_path.replace('.npy', f'{label}.npy'), rgb)
        print(f'{rgb_path} saved')
    
    
    for frame in range(1, framescount+1):
        framefolder_path = f"{data_dir}frames_{frame:04d}/"
        images = imread_with_progress([os.path.join(framefolder_path, f) for f in sorted(os.listdir(framefolder_path))])
        images = np.moveaxis(images, -1, 0).astype(np.float32)
        images = images.astype(np.float32)
        if frame in val_frames:
            print('val frame:', frame)
            print('val images:', val_idx)
            save_idx(train_idx, 'train' ,frame)
            save_idx_rgb(train_idx, 'train', frame)
            save_idx(val_idx, 'val', frame)
            save_idx_rgb(val_idx, 'val', frame)
        else:
            print(f'whole train frame:', frame)
            save_idx(all_idx, 'train', frame)
            save_idx_rgb(all_idx, 'train', frame)
        images = None
        uvst_tmp = None
    
    print('Loaded llff in', data_dir, ' hwf:', hwf)
    print(frames_folders)