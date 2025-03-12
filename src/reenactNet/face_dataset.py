import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import glob
import numpy as np
import cv2
import sys
import numpy as np


class FaceImgDataset(Dataset):

    def __init__(self, root_dir, is_test=False, aug_rotation=False, aug_random=False, aug_color=False, aug_affine_color=False, device="cuda", id="real"):
        '''
        local - eyel_img, eyer_img, lips_img
        '''
        # if id == "real":
        #     val_dir = "../../dataset/image/real/sihun_brow_test"
        # elif id == "child":
        #     val_dir = "../../dataset/image/child/normal_child"
        # elif id == "girl":
        #     val_dir = "../../dataset/image/girl/normal_girl"
        # elif id == "piers":
        #     val_dir = "../../dataset/image/piers/normal_piers"
        # elif id == "malcom":
        #     val_dir = "../../dataset/image/malcom/normal_malcom"
        val_dir = root_dir
        

        load_list = ["img_eyel", "img_eyer", "img_lips"]
        loaded_data = {}
        loaded_data_val = {}
        for load_element in load_list:
            # file_list = [glob.glob(root_dir+e, recursive=False) for e in ["/**/"+load_element+"/*.png", "/**/"+load_element+"/*.jpg"]] # match any directories, nested to any level, from the root directory
            file_list = [glob.glob(root_dir+"/"+load_element+"/*.png", recursive=False)]
            
            file_list = sum(file_list, [])
            file_list.sort()
            loaded_data[load_element] = file_list

            # file_list_val = [glob.glob(val_dir+e, recursive=False) for e in ["/**/"+load_element+"/*.png", "/**/"+load_element+"/*.jpg"]]
            file_list_val = [glob.glob(val_dir+"/"+load_element+"/*.png", recursive=False)]
            file_list_val = sum(file_list_val, [])
            file_list_val.sort()
            loaded_data_val[load_element] = file_list_val

        self.img_eyel_file_list     = loaded_data[load_list[0]]
        self.img_eyer_file_list     = loaded_data[load_list[1]]
        self.img_lips_file_list     = loaded_data[load_list[2]]

        self.img_eyel_file_list_val = loaded_data_val[load_list[0]]
        self.img_eyer_file_list_val = loaded_data_val[load_list[1]]
        self.img_lips_file_list_val = loaded_data_val[load_list[2]]
        

        assert len(self.img_eyel_file_list) == len(self.img_eyer_file_list) == len(self.img_lips_file_list)

        self.index = len(self.img_eyel_file_list)
        self.val_len = len(self.img_eyel_file_list_val)

        self.transform = {
            'base': transforms.Compose([
                                # INPUT : tensor (0~255)
                                transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.ToTensor(),
            ]),
            'random': transforms.Compose([
                                # INPUT : tensor (0~255)
                                transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                transforms.ToTensor(),
                    
            ]),
            'color': transforms.Compose([
                                # INPUT : tensor (0~255)
                                transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.ColorJitter(brightness=(0.5, 2), 
                                                        contrast=(0.5, 2), 
                                                        saturation=(0.5, 2), 
                                                        hue=(-0.2, 0.2)),
                                transforms.ToTensor(),
                    
            ]),
            'random_color': transforms.Compose([
                                # INPUT : tensor (0~255)
                                transforms.ToPILImage(),
                                transforms.Resize((128, 128)),
                                transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1)),
                                transforms.ColorJitter(brightness=(0.5, 2), 
                                                        contrast=(0.5, 2), 
                                                        saturation=(0.5, 2), 
                                                        hue=(-0.2, 0.2)),
                                transforms.ToTensor(),
            ]),

            'valid': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

        self.is_test = is_test
        self.aug_rotation = aug_rotation
        self.aug_random = aug_random
        self.aug_color = aug_color
        self.aug_affine_color = aug_affine_color
        self.device = device
        self.id = id
        print("root_dir: ", root_dir, ", Img length: ", self.index ) 

    def __len__(self):
        return self.index

    def __getitem__(self, idx):
        
        eyel_img_file_path = self.img_eyel_file_list[idx]
        eyer_img_file_path = self.img_eyer_file_list[idx]
        lips_img_file_path = self.img_lips_file_list[idx]

        idx_val = idx % self.val_len
        eyel_img_file_path_val = self.img_eyel_file_list_val[idx_val]
        eyer_img_file_path_val = self.img_eyer_file_list_val[idx_val]
        lips_img_file_path_val = self.img_lips_file_list_val[idx_val]

        file_path_list = [eyel_img_file_path, eyer_img_file_path, lips_img_file_path]
        file_path_list_val = [eyel_img_file_path_val, eyer_img_file_path_val, lips_img_file_path_val]

        out = {}

        for file_path, file_path_val in zip(file_path_list, file_path_list_val):
            name = file_path.split("/")[-2].replace("_file_path", "").replace("img_", "")
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4) # 0~255 numpy

            val_img = cv2.imread(file_path_val)
            val_img = cv2.cvtColor(val_img, cv2.COLOR_BGR2RGB)
            val_img = cv2.resize(val_img, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4)

            if self.is_test:
                ''' OUPUT : tensor (C,H,W), max value is 1 '''
                img = img.transpose(2,0,1).astype("float32")/255.0
                img = torch.from_numpy(img).to(self.device, dtype=torch.float)
                
                out[name] = {"gt": img}
            
            elif self.is_test == False: 
                ''' is_train '''
                if self.id == "mery":
                    if self.aug_rotation:
                        if name == "eyer":
                            img = self.transform['base'](img) # 0~1 tensor
                            img = TF.rotate(img, angle=-20.0) # 0~1 tensor 
                            img = img.permute(1, 2, 0).numpy() * 255.0
                    
                        if name == "eyel":
                            img = self.transform['base'](img)
                            img = TF.rotate(img, angle=20.0)
                            img = img.permute(1, 2, 0).numpy() * 255.0
                
                if self.aug_random:
                    warped_img = self.transform['random'](img.astype("uint8"))
                    warped_img = warped_img.permute(1, 2, 0).numpy() 

                if self.aug_color: # aug_color only warped_img
                    warped_img = self.transform['color'](img.astype('uint8'))
                    warped_img = warped_img.permute(1, 2, 0).numpy()

                if self.aug_affine_color:
                    warped_img = self.transform['random_color'](img.astype('uint8'))
                    warped_img = warped_img.permute(1, 2, 0).numpy()

                img = img/255.0
                val_img = val_img/255.0

                out[name] = {"gt": img, "warped": warped_img, "gt_val": val_img} # 0~1 numpy

        if self.is_test:      
            return {'gt_eyel':out["eyel"]["gt"], 'gt_eyer':out["eyer"]["gt"], 'gt_lips':out["lips"]["gt"], 'idx':idx}

        else:
            return {'gt_eyel':out["eyel"]["gt"], 's_eyel':out["eyel"]["warped"], 'v_eyel':out["eyel"]["gt_val"], \
                    'gt_eyer':out["eyer"]["gt"], 's_eyer':out["eyer"]["warped"], 'v_eyer':out["eyer"]["gt_val"], \
                    'gt_lips':out["lips"]["gt"], 's_lips':out["lips"]["warped"], 'v_lips':out["lips"]["gt_val"], \
                    'idx':idx}        



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

    def __init__(self, root_dir, anim_dir, is_test=False, aug_color=False, device="cuda"):
        
        ### animation(that rendered image frames) directory ###
            ## Blendshape case
        # self.animation = AnimationBlendShape(anim_dir)
        # val_anim_dir = "../../dataset/animation/test_normal.json"
        # self.animation_val = AnimationBlendShape(val_anim_dir)
            ## PCA case
        self.animation = AnimationPCA(anim_dir)
        # val_anim_dir = "../../dataset/animation/val_normal.pth"
        
        # val_dir = anim_dir
        # self.animation_val = AnimationPCA(val_dir)
        
        ### get full animation ###
        self.full_anim = self.animation.get_full_animation()
        # self.anim_val_full = self.animation_val.get_full_animation()

        ### image ### 
        load_list = ["img", "img_eyel", "img_eyer", "img_lips"]
        loaded_data = {}
        # loaded_data_val = {}

        for load_element in load_list:
            # file_list = [glob.glob(root_dir+e, recursive=True) for e in ["/**/"+load_element+"/*.png", "/**/"+load_element+"/*.jpg"]]
            file_list = [glob.glob(root_dir+"/"+load_element+"/*.png", recursive=False)]
            file_list = sum(file_list, [])
            file_list.sort()
            loaded_data[load_element] = file_list # contains all the file list in each list element(img, img_eyel, img_eyer, img_lips)

            # file_list_val = [glob.glob(val_dir+e, recursive=True) for e in ["/**/"+load_element+"/*.png", "/**/"+load_element+"/*.jpg"]]
            # file_list_val = [glob.glob(val_dir+"/"+load_element+"/*.png", recursive=False)]

            # file_list_val = sum(file_list_val, [])
            # file_list_val.sort()
            # loaded_data_val[load_element] = file_list_val

        self.img_file_list = loaded_data[load_list[0]]
        self.img_eyel_file_list = loaded_data[load_list[1]]
        self.img_eyer_file_list = loaded_data[load_list[2]]
        self.img_lips_file_list = loaded_data[load_list[3]]
        assert len(self.img_eyel_file_list) == len(self.img_eyer_file_list) == len(self.img_lips_file_list)
        self.index = len(self.img_eyel_file_list)

        # self.img_file_list_val = loaded_data_val[load_list[0]]
        # self.img_eyel_file_list_val = loaded_data_val[load_list[1]]
        # self.img_eyer_file_list_val = loaded_data_val[load_list[2]]
        # self.img_lips_file_list_val = loaded_data_val[load_list[3]]
        # self.val_len = len(self.img_eyel_file_list_val)
        

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
                                transforms.ColorJitter(brightness=(0.2, 2), 
                                                        contrast=(0.3, 2), 
                                                        saturation=(0.2, 2), 
                                                        hue=(-0.3, 0.3)),
                                transforms.ToTensor(),
            ])
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
        
        # idx_val = idx % self.val_len
        # img_file_path_val = self.img_file_list_val[idx_val]
        # eyel_img_file_path_val = self.img_eyel_file_list_val[idx_val]
        # eyer_img_file_path_val = self.img_eyer_file_list_val[idx_val]
        # lips_img_file_path_val = self.img_lips_file_list_val[idx_val]

        if self.is_test == False:
            parameter = self.full_anim[idx]
            # parameter_val = self.anim_val_full[idx_val]

        file_path_list = [img_file_path, eyel_img_file_path, eyer_img_file_path, lips_img_file_path]
        # file_path_list_val = [img_file_path_val, eyel_img_file_path_val, eyer_img_file_path_val, lips_img_file_path_val]
        out = {}

        # for file_path, file_path_val in zip(file_path_list, file_path_list_val):
        for file_path in file_path_list:
            name = file_path.split("/")[-2].replace("_file_path", "").replace("img_", "") # eyel, eyer, lips
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4)
            # val_img = cv2.imread(file_path_val)
            # val_img = cv2.cvtColor(val_img, cv2.COLOR_BGR2RGB)
            # val_img = cv2.resize(val_img, dsize=(128,128), interpolation=cv2.INTER_LANCZOS4)

            if self.is_test:
                ''' OUPUT : tensor (C,H,W), max value is 1 '''
                img = img.transpose(2,0,1).astype("float32")/255.0
                img = torch.from_numpy(img).to(self.device, dtype=torch.float)

                out[name] = {"gt": img}
            
            else: # train
                if self.aug_color:
                    img = self.transform['color'](img)
                    # val_img = self.transform['color'](val_img)
                warped_img, target_img, mat = random_warp(img)
                # warped_val_img, _, _ = random_warp(val_img)

                img = img / 255.0
                # val_img = val_img / 255.0

                # out[name] = {"gt": img, "warped": warped_img, "target": target_img, "warped_val": warped_val_img, "mat": mat}
                out[name] = {"gt": img, "warped": warped_img, "target": target_img, "mat": mat}


        if self.is_test:      
            return {'gt':out["img"]["gt"], 'gt_eyel':out["eyel"]["gt"], 'gt_eyer':out["eyer"]["gt"], 'gt_lips':out["lips"]["gt"], 'idx':idx}

        else:
            # return {'gt_eyel':out["eyel"]["gt"], 's_eyel':out["eyel"]["warped"], 't_eyel':out["eyel"]["target"], 'v_s_eyel':out["eyel"]["warped_val"], 'm_eyel':out["eyel"]["mat"], \
            #         'gt_eyer':out["eyer"]["gt"], 's_eyer':out["eyer"]["warped"], 't_eyer':out["eyer"]["target"], 'v_s_eyer':out["eyer"]["warped_val"], 'm_eyer':out["eyer"]["mat"], \
            #         'gt_lips':out["lips"]["gt"], 's_lips':out["lips"]["warped"], 't_lips':out["lips"]["target"], 'v_s_lips':out["lips"]["warped_val"], 'm_lips':out["lips"]["mat"], \
            #         'idx':idx, 'p':parameter, 'v_p':parameter_val}
            
            return {'gt_eyel':out["eyel"]["gt"], 's_eyel':out["eyel"]["warped"], 't_eyel':out["eyel"]["target"],  'm_eyel':out["eyel"]["mat"], \
                    'gt_eyer':out["eyer"]["gt"], 's_eyer':out["eyer"]["warped"], 't_eyer':out["eyer"]["target"],  'm_eyer':out["eyer"]["mat"], \
                    'gt_lips':out["lips"]["gt"], 's_lips':out["lips"]["warped"], 't_lips':out["lips"]["target"],  'm_lips':out["lips"]["mat"], \
                    'idx':idx, 'p':parameter, }
            

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


