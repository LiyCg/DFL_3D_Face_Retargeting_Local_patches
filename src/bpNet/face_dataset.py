import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import numpy as np
import cv2
import sys
import numpy as np

from kornia.utils.helpers import _torch_inverse_cast, _torch_solve_cast
from kornia.geometry.conversions import (
    convert_affinematrix_to_homography,
    normalize_homography,
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../dataset/data_preparation/renderer/')))
from animation import AnimationBlendShape, AnimationPCA


class FaceImgDataset_local(Dataset):

    def __init__(self, root_dir, anim_dir, is_pca = False, is_test=False, aug_color=False, device="cuda"):
        
        ### face image directory ###
        # val_dir = "../../dataset/image/child/normal_child" # child
        # val_dir = "../../dataset/image/girl/normal_girl" # girl
        val_dir = root_dir
        
        ### animation(that rendered image frames) directory ###
        if is_pca:
            ## PCA case, normalization = True as default
            self.animation = AnimationPCA(anim_dir)
            val_anim_dir = anim_dir
            self.animation_val = AnimationPCA(val_anim_dir)
            print("using pca param")
        else:
            ## Blendshape case
            self.animation = AnimationBlendShape(anim_dir)
            val_anim_dir = anim_dir
            self.animation_val = AnimationBlendShape(val_anim_dir)
            print("using bshp param")
        
        
        ### get full animation ###
        # import pdb;pdb.set_trace()
        self.full_anim = self.animation.get_full_animation()
        self.anim_val_full = self.animation_val.get_full_animation()

        ### image ### 
        load_list = ["img", "img_eyel", "img_eyer", "img_lips"]
        loaded_data = {}
        loaded_data_val = {}

        for load_element in load_list:
            # file_list = [glob.glob(root_dir+e, recursive=True) for e in ["/**/"+load_element+"/*.png", "/**/"+load_element+"/*.jpg"]]
            file_list = [glob.glob(root_dir+"/"+load_element+"/*.png", recursive=False)]
            file_list = sum(file_list, [])
            file_list.sort()
            loaded_data[load_element] = file_list # contains all the file list in each list element(img, img_eyel, img_eyer, img_lips)

            file_list_val = [glob.glob(val_dir+e, recursive=True) for e in ["/**/"+load_element+"/*.png", "/**/"+load_element+"/*.jpg"]]
            file_list_val = sum(file_list_val, [])
            file_list_val.sort()
            loaded_data_val[load_element] = file_list_val

        self.img_file_list = loaded_data[load_list[0]]
        self.img_eyel_file_list = loaded_data[load_list[1]]
        self.img_eyer_file_list = loaded_data[load_list[2]]
        self.img_lips_file_list = loaded_data[load_list[3]]
        print("img_len: ", len(self.img_eyel_file_list), "anim_len: ", self.full_anim.shape[0])
        # import pdb;pdb.set_trace()
        if is_test == False:
            assert len(self.img_eyel_file_list) == len(self.img_eyer_file_list) == len(self.img_lips_file_list) == self.full_anim.shape[0]
        self.index = len(self.img_eyel_file_list)

        self.img_file_list_val = loaded_data_val[load_list[0]]
        self.img_eyel_file_list_val = loaded_data_val[load_list[1]]
        self.img_eyer_file_list_val = loaded_data_val[load_list[2]]
        self.img_lips_file_list_val = loaded_data_val[load_list[3]]
        self.val_len = len(self.img_eyel_file_list_val)

        self.transform = {
            'base': transforms.Compose([
                                # INPUT : tensor (0~255)
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
            ]),
            'color': transforms.Compose([
                                # INPUT : tensor (0~255)
                                transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.ColorJitter(saturation=(0.1, 1)),
                                transforms.ToTensor()
            ])
#             'color': transforms.Compose([
#                                 # INPUT : tensor (0~255)
#                                 transforms.ToPILImage(),
#                                 transforms.Resize((128, 128)),
#                                 transforms.ColorJitter(brightness=(0.2, 2), 
#                                                         contrast=(0.3, 2), 
#                                                         saturation=(0.2, 2), 
#                                                         hue=(-0.3, 0.3)),
#                                 transforms.ToTensor(),
#             ])
        }

        self.is_test = is_test
        self.aug_color = aug_color
        self.device = device

        print("root_dir: ", root_dir, ", Img length: ", self.index) 

    def __len__(self):
        return self.index

    def __getitem__(self, idx):
        
        img_file_path = self.img_file_list[idx] 
        eyel_img_file_path = self.img_eyel_file_list[idx]
        eyer_img_file_path = self.img_eyer_file_list[idx]
        lips_img_file_path = self.img_lips_file_list[idx]
        
        idx_val = idx % self.val_len
        img_file_path_val = self.img_file_list_val[idx_val]
        eyel_img_file_path_val = self.img_eyel_file_list_val[idx_val]
        eyer_img_file_path_val = self.img_eyer_file_list_val[idx_val]
        lips_img_file_path_val = self.img_lips_file_list_val[idx_val]

        if self.is_test == False:
            parameter = self.full_anim[idx]
            parameter_val = self.anim_val_full[idx_val]

        file_path_list = [img_file_path, eyel_img_file_path, eyer_img_file_path, lips_img_file_path]
        file_path_list_val = [img_file_path_val, eyel_img_file_path_val, eyer_img_file_path_val, lips_img_file_path_val]
        out = {}

        for file_path, file_path_val in zip(file_path_list, file_path_list_val):
            name = file_path.split("/")[-2].replace("_file_path", "").replace("img_", "") # eyel, eyer, lips
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4)
            val_img = cv2.imread(file_path_val)
            val_img = cv2.cvtColor(val_img, cv2.COLOR_BGR2RGB)
            val_img = cv2.resize(val_img, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4)

            if self.is_test:
                ''' OUPUT : tensor (C,H,W), max value is 1 '''
                img = img.transpose(2,0,1).astype("float32")/255.0
                img = torch.from_numpy(img).to(self.device, dtype=torch.float)

                out[name] = {"gt": img}
            
            else: # train
                if self.aug_color:
                    img = self.transform['color'](img.astype("uint8"))
                    val_img = self.transform['color'](val_img.astype("uint8"))
                    img = img.permute(1, 2, 0).numpy() * 255 
                    val_img = val_img.permute(1, 2, 0).numpy() * 255
                    warped_img, target_img, mat = random_warp(img)
                    warped_val_img, _, _ = random_warp(val_img)
                else:
                    warped_img, target_img, mat = random_warp(img)
                    warped_val_img, _, _ = random_warp(val_img)

                img = img / 255.0
                val_img = val_img / 255.0

                out[name] = {"gt": img, "warped": warped_img, "target": target_img, "warped_val": warped_val_img, "mat": mat}

        if self.is_test:      
            return {'gt':out["img"]["gt"], 'gt_eyel':out["eyel"]["gt"], 'gt_eyer':out["eyer"]["gt"], 'gt_lips':out["lips"]["gt"], 'idx':idx}

        else:
            return {'gt_eyel':out["eyel"]["gt"], 's_eyel':out["eyel"]["warped"], 't_eyel':out["eyel"]["target"], 'v_s_eyel':out["eyel"]["warped_val"], 'm_eyel':out["eyel"]["mat"], \
                    'gt_eyer':out["eyer"]["gt"], 's_eyer':out["eyer"]["warped"], 't_eyer':out["eyer"]["target"], 'v_s_eyer':out["eyer"]["warped_val"], 'm_eyer':out["eyer"]["mat"], \
                    'gt_lips':out["lips"]["gt"], 's_lips':out["lips"]["warped"], 't_lips':out["lips"]["target"], 'v_s_lips':out["lips"]["warped_val"], 'm_lips':out["lips"]["mat"], \
                    'idx':idx, 'p':parameter, 'v_p':parameter_val}


class FaceImgDataset_local_test(Dataset):

    def __init__(self, root_dir, is_test=True, color=False, device="cuda"):
        
        ### image ### 
        load_list = ["img", "img_eyel", "img_eyer", "img_lips"]
        loaded_data = {}

        for load_element in load_list:
            file_list = [glob.glob(root_dir+e, recursive=True) for e in ["/**/"+load_element+"/*.png", "/**/"+load_element+"/*.jpg"]]
            file_list = sum(file_list, [])
            file_list.sort()
            loaded_data[load_element] = file_list # contains all the file list in each list element(img, img_eyel, img_eyer, img_lips)

        self.img_file_list = loaded_data[load_list[0]]
        self.img_eyel_file_list = loaded_data[load_list[1]]
        self.img_eyer_file_list = loaded_data[load_list[2]]
        self.img_lips_file_list = loaded_data[load_list[3]]
        assert len(self.img_eyel_file_list) == len(self.img_eyer_file_list) == len(self.img_lips_file_list)
        self.index = len(self.img_eyel_file_list)

        self.transform = {
            'color': transforms.Compose([
                                # INPUT : tensor (0~255)
                                transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.ColorJitter(saturation=(0.1, 1.0)),
                                # transforms.ColorJitter(brightness=(0.2, 2), 
                                #                         contrast=(0.3, 2), 
                                #                         saturation=(0.2, 2), 
                                #                         hue=(-0.3, 0.3)),
                                transforms.ToTensor(), # automatically scales the pixel intensity values of images from a range of [0,255] (typical for images stored in the standard 8-bit format) to a floating point range of [0,1].
            ])
        }
        
        self.color = color
        self.is_test = is_test
        self.device = device

        print("root_dir: ", root_dir, ", Img length: ", self.index) 

    def __len__(self):
        return self.index

    def __getitem__(self, idx):
        
        img_file_path = self.img_file_list[idx] 
        eyel_img_file_path = self.img_eyel_file_list[idx]
        eyer_img_file_path = self.img_eyer_file_list[idx]
        lips_img_file_path = self.img_lips_file_list[idx]

        file_path_list = [img_file_path, eyel_img_file_path, eyer_img_file_path, lips_img_file_path]
        out = {}

        for file_path in (file_path_list):
            name = file_path.split("/")[-2].replace("_file_path", "").replace("img_", "") # eyel, eyer, lips
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4)

            if self.is_test:
                if self.color:
                    img = self.transform['color'](img).to(self.device)
                    # img = img / 255.0
                else:
                    ''' OUPUT : tensor (C,H,W), max value is 1 '''
                    img = img.transpose(2,0,1).astype("float32")/255.0
                    img = torch.from_numpy(img).to(self.device, dtype=torch.float)

                out[name] = {"gt": img}

        if self.is_test:    
            return {'gt':out["img"]["gt"], 'gt_eyel':out["eyel"]["gt"], 'gt_eyer':out["eyer"]["gt"], 'gt_lips':out["lips"]["gt"], 'idx':idx}



def random_warp(image):
    # assert image.shape == (256, 256, 3)
    # range_ = np.zeros(7)
    w = 128
    cell_size = [ w // (2**i) for i in range(1,4) ] [ np.random.randint(3) ]
    # cell_size = 32
    cell_count = w // cell_size + 1
    # cell_count = 5
    range_ = np.linspace( 0, w, cell_count)
    # range_ = np.linspace(0, 256, 5)

    mapx = np.broadcast_to(range_, (cell_count, cell_count)).copy()
    mapy = mapx.T

    mapx = mapx + np.random.normal(size=(cell_count, cell_count)) * (cell_size*0.12)
    mapy = mapy + np.random.normal(size=(cell_count, cell_count)) * (cell_size*0.12)

    half_cell_size = cell_size // 2
    interp_mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size,half_cell_size:-half_cell_size].astype(np.float32)
    interp_mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size,half_cell_size:-half_cell_size].astype(np.float32)
    
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    
    img_torch = torch.from_numpy(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)/255)
    img_torch = torch.permute(img_torch, (2, 0, 1)).unsqueeze(0).float()

    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:w+1:cell_size, 0:w+1:cell_size].T.reshape(-1, 2)
    
    mat = umeyama(src_points, dst_points, True)[0:2]

    mat_torch = torch.from_numpy(mat).type(torch.FloatTensor).unsqueeze(0)   # [1,2,3]
    
    M_3x3: torch.Tensor = convert_affinematrix_to_homography(mat_torch)
    dst_norm_trans_src_norm: torch.Tensor = normalize_homography(M_3x3, (256, 256), (256, 256))
    mat_torch = _torch_inverse_cast(dst_norm_trans_src_norm)  # torch.Size([1, 3, 3])
     
    grid = F.affine_grid(mat_torch[:, :2, :].float(), img_torch.size(), align_corners=True)    # [1, 128, 128, 2]
    image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255).type(torch.FloatTensor).permute(2,0,1).unsqueeze(0)
    target_image = F.grid_sample(image, grid, align_corners=True)    # [1, 3, 128, 128]
    target_image =  np.array(target_image.squeeze(0).permute(1, 2, 0))
    target_image = (cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    mat_torch_2X3 = mat_torch[:, :2, :].squeeze(0)
    
    # target_image = cv2.warpAffine(image, mat, (w, w))
    # mask = cv2.warpAffine(mask, mat, (w, w))
    
    return warped_image, target_image, mat_torch_2X3



def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


class FaceLatentDataset(Dataset):

    def __init__(self, root_dir, anim_dir, device="cuda"):
        
        self.animation = AnimationPCA(anim_dir)
        self.full_anim = self.animation.get_full_animation()

        ckpt_dir_reenactNet = "../checkpoint/ReenactNet/{0}".format(root_dir)
        reenactNet_network_dir = os.path.join(ckpt_dir_reenactNet, "train_result")

        self.eyel_latents = torch.load(os.path.join(reenactNet_network_dir, "eyel_latents.pth")).squeeze(dim=1)
        self.eyer_latents = torch.load(os.path.join(reenactNet_network_dir, "eyer_latents.pth")).squeeze(dim=1)
        self.lips_latents = torch.load(os.path.join(reenactNet_network_dir, "lips_latents.pth")).squeeze(dim=1)

        self.device = device

        print("root_dir: ", reenactNet_network_dir) 

    def __len__(self):
        return len(self.eyel_latents)

    def __getitem__(self, idx):
        eyel_latent = self.eyel_latents[idx]
        eyer_latent = self.eyer_latents[idx]
        lips_latent = self.lips_latents[idx]
        parameter = self.full_anim[idx]
        return {'eyel':eyel_latent, 'eyer':eyer_latent, 'lips':lips_latent, 'p':parameter}
        # return {'p':parameter}
