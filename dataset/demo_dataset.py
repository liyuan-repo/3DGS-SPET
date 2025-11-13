import os
import cv2
import torch
import numpy as np
import mediapy as media
from torch.utils import data
from transforms3d import affines, quaternions
from misc_utils import gs_utils


class OnePoseCap_Dataset(torch.utils.data.Dataset):
    def __init__(self, obj_data_dir, num_grid_points=4096, extract_RGB=False, use_binarized_mask=False,
                 obj_database_dir=None):

        self.extract_RGB = extract_RGB
        self.obj_data_dir = obj_data_dir
        self.num_grid_points = num_grid_points
        self.obj_database_dir = obj_database_dir
        self.use_binarized_mask = use_binarized_mask

        self.arkit_box_path = os.path.join(self.obj_data_dir, 'Box.txt')
        self.arkit_pose_path = os.path.join(self.obj_data_dir, 'GT021ARposes.csv')   # 外参R,T—— tx, ty, tz, qw, qx, qy qz
        self.arkit_video_path = os.path.join(self.obj_data_dir, 'GT021_fps18.mp4')
        self.arkit_intrin_path = os.path.join(self.obj_data_dir, 'camera_K.txt')   # 内参K—— fx, fy, cx, cy

        # ### read the ARKit pose info
        with open(self.arkit_pose_path, 'r') as pf:
            self.arkit_poses = [row.strip() for row in pf.readlines() if len(row) > 0 and row[0] != '#']

        with open(self.arkit_intrin_path, 'r') as cf:
            self.arkit_camKs = [row.strip() for row in cf.readlines() if len(row) > 0 and row[0] != '#']

        # ## read the video
        if self.extract_RGB:
            RGB_dir = os.path.join(self.obj_data_dir, 'RGB')
            if not os.path.exists(RGB_dir):
                os.makedirs(RGB_dir)
                cap = cv2.VideoCapture(self.arkit_video_path)
                index = 0
                while True:
                    ret, image = cap.read()
                    if not ret:
                        break
                    cv2.imwrite(os.path.join(RGB_dir, f'{index}.png'), image)
                    index += 1
        else:
            self.video_frames = media.read_video(self.arkit_video_path)  # NxHxWx3

        # assert (len(self.arkit_poses) == len(self.arkit_camKs))

        # ### preprocess the ARKit 3D object bounding box
        with open(self.arkit_box_path, 'r') as f:
            lines = f.readlines()
            box_data = [float(e) for e in lines[1].strip().split(',')]
        ex, ey, ez = box_data[0:3]
        self.obj_bbox3d = np.array([
            [-ex, -ey, -ez],  # Front-top-left corner
            [ex, -ey, -ez],  # Front-top-right corner
            [ex, ey, -ez],  # Front-bottom-right corner
            [-ex, ey, -ez],  # Front-bottom-left corner
            [-ex, -ey, ez],  # Back-top-left corner
            [ex, -ey, ez],  # Back-top-right corner
            [ex, ey, ez],  # Back-bottom-right corner
            [-ex, ey, ez],  # Back-bottom-left corner
        ]) * 0.5
        obj_bbox3D_dims = np.array([ex, ey, ez], dtype=np.float32)
        # np.prod(obj_bbox3D_dims, axis=0) 表示bbox3D的体积，用正方体大小grid_cube_size来近似长方体bbox3D
        grid_cube_size = (np.prod(obj_bbox3D_dims, axis=0) / self.num_grid_points) ** (1 / 3)  # np.prod 返回指定轴上的乘积
        xnum, ynum, znum = np.ceil(obj_bbox3D_dims / grid_cube_size).astype(np.int64)  # np.ceil 向上取整
        xmin, ymin, zmin = self.obj_bbox3d.min(axis=0)
        xmax, ymax, zmax = self.obj_bbox3d.max(axis=0)
        zgrid, ygrid, xgrid = np.meshgrid(np.linspace(zmin, zmax, znum),
                                          np.linspace(ymin, ymax, ynum),
                                          np.linspace(xmin, xmax, xnum),
                                          indexing='ij')
        # obj_bbox3d 边界框的8个角点是{(zmin, zmax),(ymin, ymax),(xmin, xmax)}组合，且以bbox3d_position为中心
        self.bbox3d_grid_points = np.stack([xgrid, ygrid, zgrid], axis=-1).reshape(-1, 3)
        self.bbox3d_diameter = np.linalg.norm(obj_bbox3D_dims)  # 3D包围框的直径。
        # linalg=linear（线性）+algebra（代数），norm则表示范数。默认为2范数

        # bbox3d_position = np.array(box_data[0:3], dtype=np.float32)   # 通过点云计算
        # bbox3D_rot_quat = np.array(box_data[6:10], dtype=np.float32)
        # bbox3D_rot_mat = quaternions.quat2mat(bbox3D_rot_quat)
        # T_O2W = affines.compose(bbox3d_position, bbox3D_rot_mat, np.ones(3))  # object-to-world
        # 物体坐标系——世界坐标系
        # 当有旋转矩阵（R）和平移（T）时获得“齐次变换矩阵”。tfs.affines.compose 函数用于将平移、旋转和缩放参数组合成一个仿射变换矩阵。
        # 这个函数的参数通常包括平移向量（T）、旋转矩阵（R）和缩放因子（S）

        self.camKs = list()
        self.poses = list()
        self.allo_poses = list()
        self.image_IDs = list()
        Xaxis_Rmat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        # 绕X轴旋转180度 https://blog.csdn.net/qq_34885669/article/details/124344320

        camk_info = self.arkit_camKs[0]  # [time, index, fx, fy, cx, cy]
        camk_dat = [float(c) for c in camk_info.split(',')]
        camk = np.eye(3)
        camk[0, 0] = camk_dat[-4]
        camk[1, 1] = camk_dat[-3]
        camk[0, 2] = camk_dat[-2]
        camk[1, 2] = camk_dat[-1]
        self.camKs = camk

        for frame_idx, pose_info in enumerate(self.arkit_poses):
            pose_dat = [float(p) for p in pose_info.split(',')]
            bbox_pos = pose_dat[1:4]
            bbox_quat = pose_dat[4:]
            rot_mat = quaternions.quat2mat(bbox_quat)
            pose_RT = affines.compose(bbox_pos, rot_mat, np.ones(3))  # camera-to-world  [R|t]

            # T_W2C = np.linalg.inv(T_C2W)  # world-to-camera
            # pose_RT = T_W2C @ T_O2W  # object-to-camera

            allo_pose = pose_RT.copy()
            # allo_pose[:3, :3] = gs_utils.egocentric_to_allocentric(allo_pose)[:3, :3]
            self.allo_poses.append(allo_pose)
            self.poses.append(pose_RT)
            self.image_IDs.append(frame_idx)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        data_dict = dict()
        # camK = self.camKs[idx]
        pose = self.poses[idx]
        allo_pose = self.allo_poses[idx]
        image_ID = self.image_IDs[idx]

        if self.extract_RGB:
            image = cv2.imread(os.path.join(self.obj_data_dir, 'RGB', f'{image_ID}.png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        else:
            image = np.array(self.video_frames[idx]) / 255.0

        data_dict['image_ID'] = image_ID
        data_dict['camK'] = torch.as_tensor(self.camKs, dtype=torch.float32)
        data_dict['pose'] = torch.as_tensor(pose, dtype=torch.float32)
        data_dict['image'] = torch.as_tensor(image, dtype=torch.float32)
        data_dict['allo_pose'] = torch.as_tensor(allo_pose, dtype=torch.float32)

        if self.obj_database_dir is not None:
            data_dict['coseg_mask_path'] = os.path.join(self.obj_database_dir, 'pred_coseg_mask',
                                                        '{:06d}.png'.format(image_ID))
        else:
            data_dict['coseg_mask_path'] = os.path.join(self.obj_data_dir, 'pred_coseg_mask',
                                                        '{:06d}.png'.format(image_ID))

        return data_dict

    def collate_fn(self, batch):
        """
        batchify the data
        """
        new_batch = dict()
        for each_dat in batch:
            for key, val in each_dat.items():
                if key not in new_batch:
                    new_batch[key] = list()
                new_batch[key].append(val)

        for key, val in new_batch.items():
            new_batch[key] = torch.stack(val, dim=0)

        return new_batch
