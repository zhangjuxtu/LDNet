import torch.nn as nn
import cv2
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function as F
import matplotlib.pyplot as plt
from parameters import Parameters
import math
from scipy import interpolate

p = Parameters()

###############################################################
##
## visualize
## 
###############################################################

def draw_lanes(image, lanes, hsample, ratio_w, ratio_h):
    for lane in lanes:
        for index, point in enumerate(lane):
            if point > 0:
                image = cv2.circle(image, (int(point*ratio_w), int(hsample[index] * ratio_h)), 1, (0,255,255), -1)

    return image

def draw_points(x, y, image, idx=0, psize = 8):
    color_index = idx 
    for i, j in zip(x, y):
        image = cv2.circle(image, (int(i), int(j)), psize, p.color[color_index], -1)
    return image

def visualize_gt(gt_point, meta, image, p):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    hm = meta['end_hm'][0][0]

    for y in range(p.grid_y):
        for x in range(p.grid_x):
            if gt_point[0][y][x] > 0:
                #xx = int(gt_point[1][y][x]*p.resize_ratio+p.resize_ratio*x)
                #yy = int(gt_point[2][y][x]*p.resize_ratio+p.resize_ratio*y)
                xx = int(p.resize_ratio*x)
                yy = int(p.resize_ratio*y)
                image = cv2.circle(image, (xx, yy), 2, p.color[1], -1)

    fig = plt.figure()
    ax=fig.add_subplot(1, 3, 1)
    ax.imshow(image)
    ax=fig.add_subplot(1, 3, 2)
    ax.imshow(gt_point[0])
    ax=fig.add_subplot(1, 3, 3)
    ax.imshow(hm)
    plt.show()

def visualize_result(image, x, y, ratio_w, ratio_h, gt_img, masks, pred_images):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()
    image0 = deepcopy(image)

    for i, j in zip(x, y):
        #print('lane', i)
        #print('samples', j)
        for idx, pt in enumerate(i):
            if(pt == -2):
                continue
            xx = int(pt*ratio_w)
            yy = int(j[idx]*ratio_h)
            image = cv2.circle(image, (xx, yy), 2, p.color[3], -1)

    '''
    fig = plt.figure()
    ax=fig.add_subplot(2, 2, 1)
    ax.imshow(gt_img[0][0])
    ax=fig.add_subplot(2, 2, 2)
    ax.imshow(masks[0])
    ax=fig.add_subplot(2, 2, 3)
    ax.imshow(pred_images[0])
    ax=fig.add_subplot(2, 2, 4)
    ax.imshow(image)
    plt.show()
    '''
    cv2.imshow("result", image)
    cv2.waitKey(0) 

###############################################################
##
## calculate
## 
###############################################################
def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x,y):
        out_x.append((np.array(i)/ratio_w).tolist())
        out_y.append((np.array(j)/ratio_h).tolist())

    return out_x, out_y

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def sort_along_x(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(i, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def sort_batch_along_y(target_lanes, target_h):
    out_x = []
    out_y = []

    for x_batch, y_batch in zip(target_lanes, target_h):
        temp_x = []
        temp_y = []
        for x, y, in zip(x_batch, y_batch):
            ind = np.argsort(y, axis=0)
            sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
            sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
            temp_x.append(sorted_x)
            temp_y.append(sorted_y)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y

def get_pos_lane(lane, spts, epts, p):
    ys = np.linspace(spts[1], epts[1], p.points_num)
    #print('lane shape', lane.shape)
    #print('lane len:', len(lane))
    #print('lane :', lane)

    if len(lane) > 3:
        lane = np.array(sorted(lane, key=lambda x: x[1]))
        tmp_x = lane[:, 0]
        tmp_y = lane[:, 1]
        gt_tck = interpolate.splrep(tmp_y, tmp_x)
        xs = interpolate.splev(ys, gt_tck)
    elif len(lane) == 3:
        pts_num = p.points_num // 2
        xs1 = np.linspace(lane[0][0], lane[1][0],  pts_num)
        xs2 = np.linspace(lane[1][0], lane[-1][0], pts_num+1)
        xs = np.append(xs1, xs2[1:])
        ys1 = np.linspace(lane[0][1], lane[1][1], pts_num)
        ys2 = np.linspace(lane[1][1], lane[-1][1], pts_num+1)
        ys = np.append(ys1, ys2[1:])
    elif len(lane) == 2:
        assert spts[0] == lane[0][0]
        assert spts[1] == lane[0][1]
        assert epts[0] == lane[-1][0]
        assert epts[1] == lane[-1][1]
        xs = np.linspace(spts[0], epts[0], p.points_num)
    else:
        raise NotImplementedError()
    xs = np.clip(xs, 0, p.grid_x-1) 
    xys = np.stack((xs, ys), axis=1)
    return xys

#def get_init_lane(spts, epts, p):
def get_init_lane(spts, epts, gt_lane, p):
    xs = np.linspace(spts[0], epts[0], p.points_num)
    ys = np.linspace(spts[1], epts[1], p.points_num)
    xys = np.stack((xs, ys), axis=1)
    return xys

def get_init_lane_torch(spts, epts, p):
    xs = torch.linspace(float(spts[0]), float(epts[0]), p.points_num)
    ys = torch.linspace(float(spts[1]), float(epts[1]), p.points_num)
    xys = torch.stack((xs, ys), dim=1)
    return xys

# sampel the points at [0, 63, 95, 111, 127]
# sample points num [64, 32, 16, 16]
#def get_init_lane(spts, epts, gt_lane, p):
#    xs1 = np.linspace(spts[0], gt_lane[63][0], 64)
#    xs2 = np.linspace(gt_lane[64][0], gt_lane[95][0], 32)
#    xs = np.append(xs1, xs2)
#    xs3 = np.linspace(gt_lane[96][0], gt_lane[127][0], 32)
#    xs = np.append(xs, xs3)
#    #xs4 = np.linspace(gt_lane[112][0], gt_lane[127][0], 16)
#    #xs = np.append(xs, xs4)
#    ys1 = np.linspace(spts[1], gt_lane[63][1], 64)
#    ys2 = np.linspace(gt_lane[64][1], gt_lane[95][1], 32)
#    ys = np.append(ys1, ys2)
#    ys3 = np.linspace(gt_lane[96][1], gt_lane[127][1], 32)
#    ys = np.append(ys, ys3)
#    #ys4 = np.linspace(gt_lane[112][1], gt_lane[127][1], 16)
#    #ys = np.append(ys, ys4)
#    xys = np.stack((xs, ys), axis=1)
#    return xys

def gaussian2D(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)



def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def make_ground_truth_point(target_lanes, target_h, p):
    target_lanes, target_h = sort_batch_along_y(target_lanes, target_h)

    batch_size = len(target_lanes)
    ground = np.zeros((batch_size, 1, p.grid_y, p.grid_x))
    startpoints = np.zeros([batch_size, 5, 2], dtype=np.float32)
    endpoints = np.zeros([batch_size, 5, 2], dtype=np.float32)
    gt_lanes = np.zeros([batch_size, 5, p.points_num, 2], dtype=np.float32)
    gt_lanes_init = np.zeros([batch_size, 5, p.points_num, 2], dtype=np.float32)
    lane_num = np.zeros([batch_size], dtype=np.int8)
    end_hm = np.zeros([batch_size, 5, p.grid_y, p.grid_x], dtype=np.float32)

    for batch_index, batch in enumerate(target_lanes):
        lane_num[batch_index] = len(batch)
        del_lane_num = 0
        for lane_index, lane in enumerate(batch):
            tmp_lane = [[x, y] for (x, y) in zip(lane, target_h[batch_index][lane_index]) if x >= 0]
            if len(tmp_lane) < 2:
                lane_num[batch_index] -= 1
                del_lane_num += 1
                continue
            else:
                tmp_lane = np.array(tmp_lane)*1.0/p.resize_ratio
                spts = tmp_lane[0]  # start from the bottom and side
                epts = tmp_lane[-1] # vanishing point
                startpoints[batch_index][lane_index-del_lane_num] = spts
                endpoints[batch_index][lane_index-del_lane_num] = epts
                draw_umich_gaussian(end_hm[batch_index][lane_index-del_lane_num], spts, 3)

                gt_lane = get_pos_lane(tmp_lane, spts, epts, p)
                init_lane = get_init_lane(spts, epts, gt_lane, p)
                gt_lanes[batch_index][lane_index-del_lane_num] = gt_lane
                gt_lanes_init[batch_index][lane_index-del_lane_num] = init_lane

                for point_index, point in enumerate(gt_lanes[batch_index][lane_index-del_lane_num]):
                    ptx = point[0]
                    pty = point[1]
                    x_index = int(ptx)
                    y_index = int(pty)
                    ground[batch_index][0][y_index][x_index] = 1.0

    end_hm = np.sum(end_hm, axis=1)
    end_hm = np.expand_dims(end_hm, 1)
    meta = {'lane_num':lane_num, 'startpoints':startpoints, 'endpoints':endpoints,
            'gt_lanes':gt_lanes, 'gt_lanes_init':gt_lanes_init, 'end_hm':end_hm}

    return ground, meta

def vpoint_detection(seg_result, lane_num):
    for row_idx in range(p.grid_y):
        tmp_pts = torch.nonzero(seg_result[row_idx])
        if len(tmp_pts) <= 1: # skip the rows which do not have segment points
            continue
        else:
            vanish_row = row_idx
            break

    vpts1 = torch.nonzero(seg_result[vanish_row]).squeeze(-1)
    vpts2 = torch.nonzero(seg_result[vanish_row+1]).squeeze(-1)
    #print('vpts1 : ', vpts1)
    #print('vpts2 : ', vpts2)
    b_vpts1 = vpts1[-1] - vpts1[0]
    
    if len(vpts2) > len(vpts1): 
        b_vpts2 = vpts2[-1] - vpts2[0]

    if b_vpts1 > 10 or len(vpts2) <= len(vpts1):
        vpx = vpts1[0] 
        vpy = vanish_row
        vstep = b_vpts1*1.0/(lane_num -1)
    else :
        vpx = vpts2[0] 
        vpy = vanish_row + 1
        vstep = b_vpts2*1.0/(lane_num - 1)

    return vpx, vpy, vstep

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_kp_hm(kp_hm, K=10):
    batch, cat, height, width = kp_hm.size()
    kp_hm = nms(kp_hm)

    topk_scores, topk_inds = torch.topk(kp_hm.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds*1.0 / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    #topk_xs = topk_xs.view(batch, -1, K).cpu().data.numpy()
    #topk_ys = topk_ys.view(batch, -1, K).cpu().data.numpy()
    #topk_scores = topk_scores.view(batch, -1, K).cpu().data.numpy()
    topk_xs = topk_xs.view(batch, -1, K)
    topk_ys = topk_ys.view(batch, -1, K)
    topk_scores = topk_scores.view(batch, -1, K)

    return topk_xs, topk_ys, topk_scores



