#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
from dla import DLASeg
from msra_resnet import get_pose_net
from snake import Evolution
import util
import matplotlib.pyplot as plt
from copy import deepcopy

####################################################################
##
## lane_detection_network
##
####################################################################
class lane_detection_network(nn.Module):
    def __init__(self, p):
        super(lane_detection_network, self).__init__()

        self.p = p

        #feature extraction
        '''
        self.layer1 = hourglass_block(128, 128)
        self.layer2 = hourglass_block(128, 128)
        self.layer3 = hourglass_block(128, 128)
        self.layer4 = hourglass_block(128, 128)
        '''
        if p.model_name == "dla":
            self.backbone = DLASeg('dla34', {'conf':1, 'hm':1},
                pretrained=True, down_ratio=self.p.resize_ratio, final_kernel=1,
                last_level=5, head_conv=256)
        elif p.model_name == "resnet":
            self.backbone = get_pose_net(34, {'conf':1, 'hm':1})

        self.gcn = Evolution(self.p)

    #####################################################
    ## Sample
    #####################################################
    def static_sample(self, meta):
        with torch.no_grad():
            lane_num = torch.from_numpy(meta['lane_num']).cuda()
            gt_pos_lanes = torch.from_numpy(meta['gt_lanes']).cuda()
            gt_pos_init = torch.from_numpy(meta['gt_lanes_init']).cuda()

            n_batch = lane_num.size(0)

            # static line proposal
            lane_init = []
            lane_target = []
            lane_sample = {}
            for bidx in range(n_batch):
                pos_lane_num = lane_num[bidx]
                tmp_pos_init = gt_pos_init[bidx][:pos_lane_num]
                tmp_pos_target = gt_pos_lanes[bidx][:pos_lane_num]

                lane_init.append(tmp_pos_init)
                lane_target.append(tmp_pos_target) 

            lane_init = torch.cat(lane_init, dim=0) 
            lane_target = torch.cat(lane_target, dim=0) 
            #print('lane_init shape', lane_init.shape)
            #print('lane_target shape', lane_target.shape)
            lane_ind = torch.cat([torch.full([lane_num[i]], i, dtype=torch.int8) for i in range(n_batch)], dim=0)
            lane_sample['lane_init'] = lane_init
            lane_sample['lane_target'] = lane_target
            lane_sample['lane_ind'] = lane_ind
            #print('lane_init:', lane_init)
            #print('lane_target:', lane_target)
        return lane_sample

                        
    # use the end points of lane lines from ground truth 
    # use the start points from predicted vanishing point
    # to match the ground truth with predicted lane 
    def half_dynamic_sample_1(self, meta, confidences):
        with torch.no_grad():
            lane_num = torch.from_numpy(meta['lane_num']).cuda()
            gt_pos_lanes = torch.from_numpy(meta['gt_lanes']).cuda()
            gt_pos_init = torch.from_numpy(meta['gt_lanes_init']).cuda()
            startpoints = torch.from_numpy(meta['startpoints']).cuda()
            n_batch = lane_num.size(0)

            # static line proposal
            lane_init = []
            lane_target = []
            lane_sample = {}
            seg_results = torch.sigmoid(confidences)
            for bidx in range(n_batch):
                seg_result = seg_results[bidx].view(self.p.grid_y, self.p.grid_x)
                seg_result[seg_result < self.p.threshold_conf] = 0
                seg_result[seg_result > self.p.threshold_conf] = 1
                #seg_res = deepcopy(seg_result).cpu().data.numpy()
                #plt.imshow(seg_res)
                #plt.show()

                pos_lane_num = lane_num[bidx]
                tmp_pos_init = gt_pos_init[bidx][:pos_lane_num]
                tmp_pos_target = gt_pos_lanes[bidx][:pos_lane_num]

                # generate the initial lane lines
                tmp_spts = startpoints[bidx][:pos_lane_num] 
                vpx, vpy, step = util.vpoint_detection(seg_result, pos_lane_num)
                for lidx in range(pos_lane_num):
                    spt = tmp_spts[lidx] 
                    vpx_i = vpx + lidx * step
                    ept = [vpx_i, vpy]
                    tmp_pos_init[lidx] = util.get_init_lane_torch(spt, ept, self.p)
                    #print('spt', spt)
                    #print('ept', ept)
                # generate the initial lane lines

                lane_init.append(tmp_pos_init)
                lane_target.append(tmp_pos_target) 

            lane_init = torch.cat(lane_init, dim=0) 
            lane_target = torch.cat(lane_target, dim=0) 
            lane_ind = torch.cat([torch.full([lane_num[i]], i, dtype=torch.int8) for i in range(n_batch)], dim=0)
            lane_sample['lane_init'] = lane_init
            lane_sample['lane_target'] = lane_target
            lane_sample['lane_ind'] = lane_ind
            #print('lane_init:', lane_init)
            #print('lane_target:', lane_target)
        return lane_sample

    # use the predicted starting points
    # use the ending points from ground truth
    # to match the ground truth with predicted lane 
    def half_dynamic_sample_2(self, meta, heatmaps):
        with torch.no_grad():
            lane_num = torch.from_numpy(meta['lane_num']).cuda()
            gt_pos_lanes = torch.from_numpy(meta['gt_lanes']).cuda()
            startpoints = torch.from_numpy(meta['startpoints']).cuda()
            endpoints = torch.from_numpy(meta['endpoints']).cuda()
            n_batch = lane_num.size(0)

            lane_init = []
            lane_target = []
            lane_sample = {}

            kp_heatmaps = torch.sigmoid(heatmaps)
            kp_num = 8
            xs, ys, scores = util.decode_kp_hm(kp_hm=kp_heatmaps, K=kp_num)
            #kp_res = deepcopy(kp_heatmaps).cpu().data.numpy()
            #plt.imshow(kp_res[0][0])
            #plt.show()
            for bidx in range(n_batch):
                '''
                lnum = 0
                spts = []
                for j2 in range(kp_num):
                    if scores[bidx][0][j2] > 0.3:
                        lnum = lnum + 1
                        spts.append([xs[bidx][0][j2], ys[bidx][0][j2]]) 
                '''
                xs_ = xs[bidx][0]
                ys_ = ys[bidx][0]
                sc_ = scores[bidx][0]
                #xs_ = xs_[sc_ > 0.3]
                #ys_ = ys_[sc_ > 0.3]
                xys = torch.cat([xs_[..., None], ys_[..., None]], dim=-1)
                #xys = xys[..., None, :]

                pos_lane_num = lane_num[bidx]
                gt_spts = startpoints[bidx][:pos_lane_num] 
                gt_epts = endpoints[bidx][:pos_lane_num] 
                gt_spts_ = gt_spts[..., None, :]
                dist = torch.sum((gt_spts_ - xys) ** 2, -1) ** 0.5
                cost, match = torch.min(dist, -1)

                tmp_pos_init = []
                tmp_pos_target = []
                lnum = 0
                for lidx in range(pos_lane_num):
                    if cost[lidx] < self.p.pt_acc_thr:
                        lnum = lnum + 1
                        spt = xys[match[lidx]].squeeze(0)
                        ept = gt_epts[lidx] 
                        tmp_init_lane = util.get_init_lane_torch(spt, ept, self.p)
                        tmp_pos_init.append(tmp_init_lane.unsqueeze(0))
                        tmp_pos_target.append(gt_pos_lanes[bidx][lidx].unsqueeze(0))

                tmp_pos_init = torch.cat((tmp_pos_init), 0)
                tmp_pos_target = torch.cat((tmp_pos_target), 0)

                lane_num[bidx] = lnum
                lane_init.append(tmp_pos_init)
                lane_target.append(tmp_pos_target) 

            lane_init = torch.cat(lane_init, dim=0).cuda() 
            lane_target = torch.cat(lane_target, dim=0).cuda() 
            lane_ind = torch.cat([torch.full([lane_num[i]], i, dtype=torch.int8) for i in range(n_batch)], dim=0)
            lane_sample['lane_init'] = lane_init
            lane_sample['lane_target'] = lane_target
            lane_sample['lane_ind'] = lane_ind
            #print('lane_init:', lane_init)
            #print('lane_target:', lane_target)
        return lane_sample

    # use the predicted starting points
    # use the ending points from ground truth
    # to match the ground truth with predicted lane 
    def half_dynamic_sample_3(self, meta, heatmaps):
        with torch.no_grad():
            lane_num = torch.from_numpy(meta['lane_num']).cuda()
            gt_pos_lanes = torch.from_numpy(meta['gt_lanes']).cuda()
            gt_pos_init = torch.from_numpy(meta['gt_lanes_init']).cuda()
            startpoints = torch.from_numpy(meta['startpoints']).cuda()
            endpoints = torch.from_numpy(meta['endpoints']).cuda()
            n_batch = lane_num.size(0)

            lane_init = []
            lane_target = []
            lane_sample = {}

            kp_heatmaps = torch.sigmoid(heatmaps)
            kp_num = 8
            xs, ys, scores = util.decode_kp_hm(kp_hm=kp_heatmaps, K=kp_num)
            #kp_res = deepcopy(kp_heatmaps).cpu().data.numpy()
            #plt.imshow(kp_res[0][0])
            #plt.show()
            for bidx in range(n_batch):
                '''
                lnum = 0
                spts = []
                for j2 in range(kp_num):
                    if scores[bidx][0][j2] > 0.3:
                        lnum = lnum + 1
                        spts.append([xs[bidx][0][j2], ys[bidx][0][j2]]) 
                '''
                xs_ = xs[bidx][0]
                ys_ = ys[bidx][0]
                sc_ = scores[bidx][0]
                xs_ = xs_[sc_ > 0.3]
                ys_ = ys_[sc_ > 0.3]
                xys = torch.cat([xs_[..., None], ys_[..., None]], dim=-1)
                xys_ = xys[..., None, :]

                pos_lane_num = lane_num[bidx]
                gt_spts = startpoints[bidx][:pos_lane_num] 
                gt_epts = endpoints[bidx][:pos_lane_num] 
                #gt_spts_ = gt_spts[..., None, :]
                #dist = torch.sum((gt_spts_ - xys) ** 2, -1) ** 0.5
                dist = torch.sum((xys_ - gt_spts) ** 2, -1) ** 0.5
                cost, match = torch.min(dist, -1)

                #tmp_pos_init = gt_pos_init[bidx][:pos_lane_num]
                tmp_pos_init = []
                lnum = 0
                #for lidx in range(pos_lane_num):
                for lidx in range(len(xys)):
                    if cost[lidx] < self.p.pt_acc_thr:
                        lnum = lnum + 1
                        spt = xys[lidx]
                        ept = gt_epts[match[lidx]] 
                        tmp_init_lane = util.get_init_lane_torch(spt, ept, self.p)
                        tmp_pos_init.append(tmp_init_lane.unsqueeze(0))

                lane_num[bidx] = lnum
                if lnum ==0:
                    continue
                tmp_pos_init = torch.cat((tmp_pos_init), 0)
                lane_init.append(tmp_pos_init)

            lane_init = torch.cat(lane_init, dim=0).cuda() 
            lane_ind = torch.cat([torch.full([lane_num[i]], i, dtype=torch.int8) for i in range(n_batch)], dim=0)
            lane_sample['lane_init'] = lane_init
            lane_sample['lane_ind'] = lane_ind
            #print('lane_init:', lane_init)
            #print('lane_target:', lane_target)
        return lane_sample

    # use the end points from heatmap
    # use the start points from predicted vanishing point
    # just used for test stage
    def full_dynamic_sample_1(self, heatmaps, confidences, meta):
        with torch.no_grad():
            lane_num = torch.from_numpy(meta['lane_num']).cuda()
            gt_pos_lanes = torch.from_numpy(meta['gt_lanes']).cuda()
            startpoints = torch.from_numpy(meta['startpoints']).cuda()
            #endpoints = torch.from_numpy(meta['endpoints']).cuda()
            n_batch = lane_num.size(0)

            lane_init = []
            lane_target = []
            lane_sample = {}

            seg_results = torch.sigmoid(confidences)
            kp_heatmaps = torch.sigmoid(heatmaps)
            kp_num = 8
            xs, ys, scores = util.decode_kp_hm(kp_hm=kp_heatmaps, K=kp_num)
            #kp_res = deepcopy(kp_heatmaps).cpu().data.numpy()
            #plt.imshow(kp_res[0][0])
            #plt.show()
            for bidx in range(n_batch):
                seg_result = seg_results[bidx].view(self.p.grid_y, self.p.grid_x)
                seg_result[seg_result < self.p.threshold_conf] = 0
                seg_result[seg_result > self.p.threshold_conf] = 1

                xs_ = xs[bidx][0]
                ys_ = ys[bidx][0]
                sc_ = scores[bidx][0]
                #xs_ = xs_[sc_ > 0.3]
                #ys_ = ys_[sc_ > 0.3]
                xys = torch.cat([xs_[..., None], ys_[..., None]], dim=-1)
                #xys = xys[..., None, :]

                pos_lane_num = lane_num[bidx]
                gt_spts = startpoints[bidx][:pos_lane_num] 
                #gt_epts = endpoints[bidx][:pos_lane_num] 
                gt_spts_ = gt_spts[..., None, :]
                dist = torch.sum((gt_spts_ - xys) ** 2, -1) ** 0.5
                cost, match = torch.min(dist, -1)

                vpx, vpy, step = util.vpoint_detection(seg_result, pos_lane_num)

                tmp_pos_init = []
                tmp_pos_target = []
                lnum = 0
                for lidx in range(pos_lane_num):
                    #if cost[lidx] < self.p.pt_acc_thr:
                    lnum = lnum + 1
                    spt = xys[match[lidx]].squeeze(0)
                    vpx_i = vpx + lidx * step
                    ept = [vpx_i, vpy]
                    #ept = gt_epts[lidx] 
                    tmp_init_lane = util.get_init_lane_torch(spt, ept, self.p)
                    tmp_pos_init.append(tmp_init_lane.unsqueeze(0))
                    tmp_pos_target.append(gt_pos_lanes[bidx][lidx].unsqueeze(0))

                tmp_pos_init = torch.cat((tmp_pos_init), 0)
                tmp_pos_target = torch.cat((tmp_pos_target), 0)

                lane_num[bidx] = lnum
                lane_init.append(tmp_pos_init)
                lane_target.append(tmp_pos_target) 

            lane_init = torch.cat(lane_init, dim=0).cuda() 
            lane_target = torch.cat(lane_target, dim=0).cuda() 
            lane_ind = torch.cat([torch.full([lane_num[i]], i, dtype=torch.int8) for i in range(n_batch)], dim=0)
            lane_sample['lane_init'] = lane_init
            lane_sample['lane_target'] = lane_target
            lane_sample['lane_ind'] = lane_ind
            #print('lane_init:', lane_init)
            #print('lane_target:', lane_target)
        return lane_sample


    # use the end points from heatmap
    # use the start points from predicted vanishing point
    # just used for test stage
    def full_dynamic_sample_2(self, heatmaps, confidences, meta):
        with torch.no_grad():

            n_batch = heatmaps.size(0)
            lane_num = torch.zeros(n_batch, dtype=torch.uint8).cuda()
            lane_init = []
            lane_target = []
            lane_sample = {}
            seg_results = torch.sigmoid(confidences)
            kp_heatmaps = torch.sigmoid(heatmaps)
            kp_num = 8
            xs, ys, scores = util.decode_kp_hm(kp_hm=kp_heatmaps, K=kp_num)
            #print('scores: ', scores)
            for bidx in range(n_batch):
                seg_result = seg_results[bidx].view(self.p.grid_y, self.p.grid_x)
                seg_result[seg_result < self.p.threshold_conf] = 0
                seg_result[seg_result > self.p.threshold_conf] = 1
                #seg_res = deepcopy(seg_result).cpu().data.numpy()
                #plt.imshow(seg_res)
                #plt.show()

                #generate the endpoints and number of lanes
                #the score of ednpoint should high enough
                #and the corresponding value on the segmentation map should high enough
                lnum = 0
                spts = []
                for j2 in range(kp_num):
                    #if scores[bidx][0][j2] > 0.35:
                    #    lnum = lnum + 1
                    #    spts.append([xs[bidx][0][j2], ys[bidx][0][j2]]) 
                    if scores[bidx][0][j2] < 0.10:
                        continue
                    for j2_tmp in range(j2+1, kp_num):
                        tmp_dist = ((xs[bidx][0][j2] - xs[bidx][0][j2_tmp])**2 + (ys[bidx][0][j2] - ys[bidx][0][j2_tmp])**2)
                        if tmp_dist < 100:
                            scores[bidx][0][j2_tmp] = 0
                    spts.append([xs[bidx][0][j2], ys[bidx][0][j2]]) 
                    lnum = lnum + 1

                spts = sorted(spts, key=lambda x:x[0])
                for i in range(len(spts) - 1):
                    if abs(spts[i][0] - spts[i+1][0]) < 3:
                        if spts[i][0] <20 and  spts[i][1] > spts[i+1][1]:
                            tmp_y = spts[i][1]
                            spts[i][1] = spts[i+1][1]
                            spts[i+1][1] = tmp_y 
                        if spts[i][0] > self.p.grid_x-20 and  spts[i][1] < spts[i+1][1]:
                            tmp_y = spts[i][1]
                            spts[i][1] = spts[i+1][1]
                            spts[i+1][1] = tmp_y 

                #detect the vanishing line
                vpx, vpy, step = util.vpoint_detection(seg_result, lnum)
                tmp_pos_init = []
                for lidx in range(lnum):
                    spt = spts[lidx] 
                    vpx_i = vpx + lidx * step
                    ept = [vpx_i, vpy]
                    tmp_init_lane = util.get_init_lane_torch(spt, ept, self.p)
                    tmp_pos_init.append(tmp_init_lane.unsqueeze(0))

                lane_num[bidx] = lnum
                if lnum == 0:
                    continue

                tmp_pos_init = torch.cat((tmp_pos_init), 0)
                lane_init.append(tmp_pos_init)

            #lane_init = torch.cat(lane_init, dim=0) 
            lane_init = torch.cat(lane_init, dim=0).cuda() 
            lane_ind = torch.cat([torch.full([lane_num[i]], i, dtype=torch.int8) for i in range(n_batch)], dim=0)
            lane_sample['lane_init'] = lane_init
            lane_sample['lane_ind'] = lane_ind
        return lane_sample

    def forward(self, inputs, meta=None, nepoch=50):
        #feature extraction
        result = self.backbone(inputs)
        heatmaps = result['hm'] 
        confidence = result['conf']
        features = result['feature'] 
        confidences = [confidence]
         
        #lane_sample = self.static_sample(meta)
        if nepoch <= self.p.dynamic_sample_at:
            lane_sample = self.static_sample(meta)
        elif self.p.dynamic_sample == False and nepoch > self.p.dynamic_sample_at:
            #lane_sample = self.half_dynamic_sample_1(meta, confidence)
            if self.training:
                lane_sample = self.half_dynamic_sample_2(meta, heatmaps)
            else:
                lane_sample = self.half_dynamic_sample_3(meta, heatmaps)
        elif self.p.dynamic_sample == True and nepoch > self.p.dynamic_sample_at:
            if self.training:
                lane_sample = self.full_dynamic_sample_1(heatmaps, confidence, meta)
            else:
                lane_sample = self.full_dynamic_sample_2(heatmaps, confidence, meta)

        # features size : [batch, 128, 64, 128] 
        # print('features shape', features.shape)
        lane_preds = self.gcn(lane_sample, features)

        return confidences, heatmaps, lane_sample, lane_preds  
