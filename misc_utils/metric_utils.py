import numpy as np


# def get_score(targets, ori_pred, pos_pred):
def get_score(pred_RT, gt_RT):
    """Score definition:
    https://kelvins.esa.int/satellite-pose-estimation-challenge/scoring/
    https://arxiv.org/abs/1911.02050
    """
    pred_R = pred_RT[:3, :3]
    pred_t = pred_RT[:3, 3]
    gt_R = gt_RT[:3, :3]
    gt_t = gt_RT[:3, 3]

    # 1. Translation error (e_t):
    t_error = np.linalg.norm(pred_t - gt_t)

    # 2. Normalized Translation error (e_t/t)
    norm_t_error = t_error / np.linalg.norm(gt_t)

    # 3. Rotation error (e_q)
    inter_sum = np.abs(np.sum(pred_R * gt_R, axis=1, keepdims=True))
    # Scaling down intermediate sum to avoid nan of arccos(x) when x > 1 (seems it is what ESA does for scoring) :
    # Set it to one when the sum is just above 1 : I guess the overflow is due to numerical errors
    # Raise Value Error when it is greater than 1.01 : the overflow is due to errors in model prediction
    if True in inter_sum[inter_sum > 1.01]:
        raise ValueError("Intermediate sum issue due to error in model prediction (orientation)")
        # Remark: it seems that ESA scoring (website) is not Raising error but scaling down to zero.
        # In practice this condition is true only when there are issues with the model or loss function.
        # With the current code this error was never raised
        # inter_sum[inter_sum > 1.01] = 0
    inter_sum[inter_sum > 1] = 1

    mean_R_error = np.mean(2 * np.arccos(inter_sum))
    mean_R_error_deg = mean_R_error * 180 / np.pi

    # 4. ESA score
    if mean_R_error_deg < 0.169 and norm_t_error < 0.002173:
        esa_score = 0
    else:
        esa_score = mean_R_error + norm_t_error

    return t_error, mean_R_error_deg,  esa_score


def calc_pose_error(pred_RT, gt_RT, unit='m'):
    pred_R = pred_RT[:3, :3]
    pred_t = pred_RT[:3, 3]
    gt_R = gt_RT[:3, :3]
    gt_t = gt_RT[:3, 3]
    R_err = np.arccos(np.clip(np.trace(pred_R @ gt_R.T) / 2 - 0.5, -1, 1.0)) / np.pi * 180
    t_err = np.linalg.norm(pred_t - gt_t)

    if unit == 'm':
        unit_factor = 1
    elif unit == 'cm':
        unit_factor = 100
    elif unit == 'mm':
        unit_factor = 1000
    else:
        raise ValueError('Invalid unit')

    t_err *= unit_factor
    return R_err, t_err


def calc_add_metric(model_3D_pts, diameter, pose_pred, pose_target, percentage=0.1, return_error=False, syn=False,
                    model_unit='m'):
    from scipy import spatial
    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_target.shape[0] == 4:
        pose_target = pose_target[:3]

    diameter_thres = diameter * percentage
    model_pred = np.dot(model_3D_pts, pose_pred[:, :3].T) + pose_pred[:, 3]
    model_target = np.dot(model_3D_pts, pose_target[:, :3].T) + pose_target[:, 3]

    if syn:
        mean_dist_index = spatial.cKDTree(model_pred)
        mean_dist, _ = mean_dist_index.query(model_target, k=1)
        mean_dist = np.mean(mean_dist)
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_target, axis=-1))

    if return_error:
        return mean_dist
    elif mean_dist < diameter_thres:
        return True
    else:
        return False


def calc_projection_2d_error(model_3D_pts, pose_pred, pose_targets, K, pixels=5, return_error=True):
    def project(xyz, K, RT):
        """
        NOTE: need to use original K
        xyz: [N, 3]
        K: [3, 3]
        RT: [3, 4]
        """
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy

    # Dim check:
    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_targets.shape[0] == 4:
        pose_targets = pose_targets[:3]

    model_2d_pred = project(model_3D_pts, K, pose_pred)  # pose_pred: 3*4
    model_2d_targets = project(model_3D_pts, K, pose_targets)
    proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
    if return_error:
        return proj_mean_diff
    elif proj_mean_diff < pixels:
        return True
    else:
        return False


def calc_bbox_IOU(pd_bbox, gt_bbox, iou_threshold=0.5, return_iou=False):
    px1, py1, px2, py2 = pd_bbox.squeeze()
    gx1, gy1, gx2, gy2 = gt_bbox.squeeze()
    inter_wid = np.maximum(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0)
    inter_hei = np.maximum(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0)
    inter_area = inter_wid * inter_hei
    outer_wid = np.maximum(px2, gx2) - np.minimum(px1, gx1)
    outer_hei = np.maximum(py2, gy2) - np.minimum(py1, gy1)
    outer_area = outer_wid * outer_hei
    iou = inter_area / outer_area
    if return_iou:
        return iou
    elif iou > iou_threshold:
        return True
    else:
        return False


def aggregate_metrics(metrics, pose_thres=[1, 3, 5], proj2d_thres=5):
    """ Aggregate metrics for the whole dataset:
    (This method should be called once per dataset)
    """
    R_errs = metrics["R_errs"]
    t_errs = metrics["t_errs"]

    agg_metric = {}
    for pose_threshold in pose_thres:
        agg_metric[f"{pose_threshold}˚@{pose_threshold}cm"] = np.mean(
            (np.array(R_errs) < pose_threshold) & (np.array(t_errs) < pose_threshold)
        )
        agg_metric[f"{pose_threshold}cm"] = np.mean((np.array(t_errs) < pose_threshold))
        agg_metric[f"{pose_threshold}˚"] = np.mean((np.array(R_errs) < pose_threshold))

    if "ADD_metric" in metrics:
        ADD_metric = metrics['ADD_metric']
        agg_metric["ADD"] = np.mean(ADD_metric)

    if "Proj2D" in metrics:
        proj2D_metric = metrics['Proj2D']
        agg_metric[f"pix@{proj2d_thres}"] = np.mean(np.array(proj2D_metric) < proj2d_thres)

    return agg_metric
