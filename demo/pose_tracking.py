import os
import time
import cv2
import sys
import torch
import pickle
import numpy as np
from misc_utils import gs_utils
from pandas import read_csv, DataFrame
from argparse import ArgumentParser
import torch.nn.functional as torch_F
from config import inference_cfg as CFG
from dataset.SPARK_dataset import SPARK_Dataset
from gaussian_object import ModelParams, GaussianModel
from gaussian_object.arguments import PipelineParams, OptimizationParams
from gaussian_object.build_3DGaussianObject import create_3D_Gaussian_object
from inference import create_reference_database_from_RGB_images, perform_segmentation_and_encoding, \
    multiple_initial_pose_inference, multiple_refine_pose_with_GS_refiner, GS_Tracker
from transforms3d import affines, quaternions
from misc_utils.metric_utils import calc_pose_error
from model.network import model_arch as ModelNet

PROJ_ROOT = os.path.dirname(os.getcwd())
sys.path.append(PROJ_ROOT)

# img0_path = "../GT021/img001_GT021.png"

zoom_image_margin = 0
zoom_image_scale = 224

if __name__ == '__main__':
    ckpt_file = os.path.join(PROJ_ROOT, 'checkpoints/model_weights.pth')
    device = torch.device('cuda:0')
    model_net = ModelNet().to(device)
    model_net.load_state_dict(torch.load(ckpt_file, map_location=device))
    model_net.eval()
    print('Model weights are loaded!')

    # -----------------------1. Capture a new object-----------------------------
    obj_name = 'RT509'  # 物体名称

    demo_data_dir = os.path.abspath(os.path.join(PROJ_ROOT, 'SPARK2024', obj_name))  # demo数据目录
    refer_seq_dir = os.path.join(demo_data_dir, f'{obj_name}_refer')  # 参考序列目录 {obj_name}_refer
    query_seq_dir = os.path.join(demo_data_dir, f'{obj_name}_test')  # 查询序列目录 {obj_name}_test

    test_pose_path = os.path.join(demo_data_dir, f'{obj_name}_test.csv')
    # refer_image_path = os.path.join(refer_seq_dir, 'RT509_refer')
    arkit_intrin_path = os.path.join(demo_data_dir, 'camera_K.txt')
    query_imgs = os.listdir(query_seq_dir)
    query_imgs.sort()
    num_frames = len(query_imgs)

    pose_track_path = os.path.join(demo_data_dir, f'{obj_name}_tracking')
    if not os.path.exists(pose_track_path):
        os.makedirs(pose_track_path)

    camKs = list()
    with open(arkit_intrin_path, 'r') as cf:
        arkit_camKs = [row.strip() for row in cf.readlines() if len(row) > 0 and row[0] != '#']
    camk_info = arkit_camKs[0]  # [time, index, fx, fy, cx, cy]
    camk_dat = [float(c) for c in camk_info.split(',')]
    camk = np.eye(3)
    camk[0, 0] = camk_dat[-4]
    camk[1, 1] = camk_dat[-3]
    camk[0, 2] = camk_dat[-2]
    camk[1, 2] = camk_dat[-1]
    camKs.append(camk)

    # ---------2. Create reference database using object reference video---------
    obj_refer_dataset = None  # 参考物体数据集
    obj_database_dir = os.path.join(demo_data_dir, f'{obj_name}-database')  # 物体数据库目录 /demo_cam/database
    obj_database_path = os.path.join(obj_database_dir, f'refer_database_{obj_name}.pkl')
    # 目标物体数据库路径 obj_database_path-> /demo_cam/database/reference_database.pkl

    if not os.path.exists(obj_database_path):
        print(f'Generate object reference database for {obj_name} ...')

        obj_refer_dataset = SPARK_Dataset(obj_data_dir=demo_data_dir,
                                          obj_name=obj_name,
                                          obj_database_dir=obj_database_dir,
                                          use_binarized_mask=CFG.BINARIZE_MASK)  # 不使用二值化的mask

        # (1) extract the object semantic tokens (Fobj) from the keyframe feature tokens (Ffps)
        # (2) employ an RA-Encoder to extract the rotation-aware embeddings {Viobj}N i=1 r from the segmented images
        reference_database = create_reference_database_from_RGB_images(model_net,
                                                                       obj_refer_dataset,
                                                                       save_pred_mask=True, device=device)

        obj_bbox3D = torch.as_tensor(obj_refer_dataset.obj_bbox3d, dtype=torch.float32)  # 3D目标框的8个角点
        bbox3d_diameter = torch.as_tensor(obj_refer_dataset.bbox3d_diameter, dtype=torch.float32)
        reference_database['obj_bbox3D'] = obj_bbox3D  # 3D目标检测框(8个角点)
        reference_database['bbox3d_diameter'] = bbox3d_diameter  # 检测框直径

        parser = ArgumentParser(description="Training script parameters")
        # ##### 参数 arguments for 3D-Gaussian Splatting Refiner ########
        gaussian_ModelP = ModelParams(parser)
        gaussian_PipeP = PipelineParams(parser)
        gaussian_OptimP = OptimizationParams(parser)
        gaussian_BG = torch.zeros(3, device=device)

        if 'ipykernel_launcher.py' in sys.argv[0]:
            args = parser.parse_args(sys.argv[3:])  # if run in ipython notebook
        else:
            args = parser.parse_args()  # if run in terminal

        print(f'Creating 3D-OGS model for {obj_name} ')
        gs_pipeData = gaussian_PipeP.extract(args)
        gs_modelData = gaussian_ModelP.extract(args)
        gs_optimData = gaussian_OptimP.extract(args)

        gs_modelData.model_path = obj_database_dir
        gs_modelData.referloader = obj_refer_dataset
        gs_modelData.queryloader = obj_refer_dataset

        # (3) create a 3D Gaussian Object representation Gobj (viewed as a 3D point cloud) for pose refinement
        # using all segmented images with poses. ie. all segmented reference images with known poses are utilized
        # to build 3DGS model
        obj_gaussians = create_3D_Gaussian_object(gs_modelData, gs_optimData, gs_pipeData, return_gaussian=True)
        # reference_database = {'obj_gaussians_path': f'{obj_database_dir}/3DGO_model.ply'}  # 错误的
        reference_database['obj_gaussians_path'] = f'{obj_database_dir}/3DGO_model_{obj_name}.ply'

        for _key, _val in reference_database.items():
            # isinstance() 函数来判断一个对象是否是一个已知的类型
            if isinstance(_val, torch.Tensor):
                reference_database[_key] = _val.detach().cpu().numpy()
        # detach意为分离，对某个张量调用函数detach()的作用是返回一个Tensor，它和原张量的数据相同，但requires_grad=False，
        # 也就意味着detach()得到的张量不会具有梯度
        with open(obj_database_path, 'wb') as file:
            pickle.dump(reference_database, file)  # 使用pickle.dump函数可以将数据序列化并保存到文件中
        print('save database to ', obj_database_path)
    else:
        print('Load database from ', obj_database_path)
        with open(obj_database_path, 'rb') as df:
            reference_database = pickle.load(df)

    for _key, _val in reference_database.items():
        if isinstance(_val, np.ndarray):
            reference_database[_key] = torch.as_tensor(_val, dtype=torch.float32).to(device)

    gs_ply_path = reference_database['obj_gaussians_path']  # ./3DGO_model.ply
    obj_gaussians = GaussianModel(sh_degree=3)
    obj_gaussians.load_ply(gs_ply_path)
    print('load 3D-OGS model from ', gs_ply_path)
    reference_database['obj_gaussians'] = obj_gaussians
    obj_bbox3d = reference_database['obj_bbox3D'].cpu()
    # 1）创建和加载 reference_database
    # 2）加载 3D-OGS model

    # -------------3. Load test data for pose estimaton and tracking-----------------
    print('Load test data for pose estimaton and tracking')
    poses = list()
    bbox_pose = list()
    gsp_accum_runtime = 0
    all_errors = list()
    track_accum_runtime = 0

    # --------------estimate the pose for the first frame---------------
    start_idx = 0
    frame_interval = 1
    camK0 = camKs[0]  # 提取每一帧的相机参数
    camK0 = torch.as_tensor(camK0, dtype=torch.float32)  # 相机参数转换为tensor类型
    img_path = os.path.join(query_seq_dir, query_imgs[start_idx])
    image_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image0 = torch.as_tensor(image_ori, dtype=torch.float32) / 255.0

    # ------------------调整视频帧的大小---------------------
    raw_hei, raw_wid = image0.shape[:2]  # 1920×1440
    raw_long_size = max(raw_hei, raw_wid)  # 1920 高
    raw_short_size = min(raw_hei, raw_wid)  # 1440 宽
    raw_aspect_ratio = raw_short_size / raw_long_size  # 原始高宽比/纵横比 Aspect Ratio  1440/1920=0.75
    if raw_hei < raw_wid:
        new_wid = CFG.query_longside_scale  # query_longside_scale = 672  长边尺寸
        new_hei = int(new_wid * raw_aspect_ratio)
    else:
        new_hei = CFG.query_longside_scale  # new_hei=672
        new_wid = int(new_hei * raw_aspect_ratio)  # new_wid=672*0.75=504

    # 将原始帧的大小，由1920×1440调整为672*504，缩放因子为 672/1920=0.35
    query_rescaling_factor = CFG.query_longside_scale / raw_long_size  # 672/1920=0.35
    que_image = image0[None, ...].permute(0, 3, 1, 2).to(device)
    que_image = torch_F.interpolate(que_image, size=(new_hei, new_wid), mode='bilinear', align_corners=True)

    # 执行segmentation and encoding的查询图像固定为672*504
    obj_data = perform_segmentation_and_encoding(model_net, que_image, reference_database, device=device)
    obj_data['camK'] = camK0.to(device)
    obj_data['img_scale'] = max(image0.shape[:2])
    obj_data['bbox_scale'] /= query_rescaling_factor  # back to the original image scale
    obj_data['bbox_center'] /= query_rescaling_factor  # back to the original image center

    init_pose = multiple_initial_pose_inference(obj_data, ref_database=reference_database, device=device)[0]
    track_pose = init_pose.copy()

    print('Perform pose tracking')
    # ------------ test image poses----------------
    test_poses = read_csv(test_pose_path)  # -----read the query seq pose info-----
    for frame_idx in range(start_idx, num_frames, frame_interval):
        pose_dat = test_poses.iloc[frame_idx]
        gt_pos = [pose_dat["Tx"], pose_dat["Ty"], pose_dat["Tz"]]
        gt_quat = [pose_dat["Qw"], pose_dat["Qx"], pose_dat["Qy"], pose_dat["Qz"]]
        rot_mat = quaternions.quat2mat(gt_quat)
        gt_pose = affines.compose(gt_pos, rot_mat, np.ones(3))  # [R|t]

        # --------- test image load-----------
        img_path = os.path.join(query_seq_dir, query_imgs[frame_idx])
        image_ori = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image0 = torch.as_tensor(image_ori, dtype=torch.float32) / 255.0
        image_hei, image_wid = image0.shape[:2]

        # ----------Perform  GS_Tracker--------------
        track_timer = time.time()

        track_outp = GS_Tracker(model_net, frame=image0, prev_pose=track_pose,
                                camK=camK0, ref_database=reference_database)
        frame_cost = time.time() - track_timer
        track_accum_runtime += frame_cost

        iter_step = track_outp['iter_step']
        render_img = track_outp['render_img']
        track_pose = track_outp['track_pose']
        bbox_scale = track_outp['bbox_scale']
        bbox_center = track_outp['bbox_center']

        # -----------------track bbox3d-------------------------------
        track_RT = torch.as_tensor(track_pose, dtype=torch.float32)
        track_bbox_KRT = torch.einsum('ij,kj->ki', track_RT[:3, :3], obj_bbox3d) + track_RT[:3, 3][None, :]
        track_bbox_KRT = torch.einsum('ij,kj->ki', camK0, track_bbox_KRT)
        track_bbox_pts = (track_bbox_KRT[:, :2] / track_bbox_KRT[:, 2:3]).type(torch.int64)
        track_bbox_pts = track_bbox_pts.numpy()

        # -----------------GT bbox3d-------------------------------
        gt_RT = torch.as_tensor(gt_pose, dtype=torch.float32)  # 外参矩阵[R|t] 4×4
        obj_bbox3d = torch.as_tensor(obj_bbox3d, dtype=torch.float32)
        # 将世界坐标系中的点Pw变换到相机坐标系中，即P=R×Pw+t
        # 矩阵操作万能函数 torch.einsum()
        gt_bbox_KRT = torch.einsum('ij,kj->ki', gt_RT[:3, :3], obj_bbox3d) + gt_RT[:3, 3][None, :]
        # 将相机坐标系中的点P变换到像素坐标系，即Z[u,v,1]=KP
        gt_bbox_KRT = torch.einsum('ij,kj->ki', camK0, gt_bbox_KRT)
        gt_bbox_pts = (gt_bbox_KRT[:, :2] / gt_bbox_KRT[:, 2:3]).type(torch.int64)
        # gsp_bbox_pts = gsp_bbox_KRT[x/z, y/z] 坐标齐次化, depth = gsp_bbox_KRT[:, 2:3]
        gt_bbox_pts = gt_bbox_pts.numpy()  # 3D边界框的8个角点。

        # ------------------------bbox3D visualization-------------------------
        color_3dgs = (255, 0, 255)  # ours
        color_gt = (0, 255, 255)  # 黄色

        query_img_np = image_ori.astype(np.uint8)
        query_bbox3d_frame = query_img_np.copy()
        track_bbox3d_img = gs_utils.draw_3d_bounding_box(query_bbox3d_frame, track_bbox_pts,
                                                         color=color_3dgs, linewidth=5)
        mix_bbox3d_frame = gs_utils.draw_3d_bounding_box(track_bbox3d_img, gt_bbox_pts,
                                                         color=color_gt, linewidth=12)

        pose_track_img = os.path.join(pose_track_path, f'{query_imgs[frame_idx]}')
        cv2.imwrite(pose_track_img, mix_bbox3d_frame)

        R_error_deg, T_error = np.round(calc_pose_error(track_pose, gt_pose), 2)
        # print(R_error_deg, T_error)
        print(frame_idx)
        all_error = [R_error_deg, T_error]
        all_errors.append(all_error)
        if (frame_idx + 1) % 20 == 0:
            print(
                '[{}/{}], \t{:.1f} FPS'.format(frame_idx + 1, num_frames,
                                               (frame_idx - start_idx) / track_accum_runtime))

    filename = os.path.join(demo_data_dir, f'track_error_{obj_name}.csv')
    track_error = DataFrame(all_errors, columns=['R_error_deg, T_error'])
    track_error.to_csv(filename, index=False)  # 将DataFrame写入CSV文件，不包含索引
