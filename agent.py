#########################################################################
##
## train agent that has some utility for training and saving.
##
#########################################################################

import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from hourglass_network import lane_detection_network
from torch.autograd import Function as F
from torch.optim.lr_scheduler import MultiStepLR
from parameters import Parameters
import math
import util
from collections import OrderedDict

############################################################
##
## agent for lane detection
##
############################################################
class Agent(nn.Module):

    #####################################################
    ## Initialize
    #####################################################
    def __init__(self):
        super(Agent, self).__init__()

        self.p = Parameters()
        self.p.device = torch.device('cuda:{}'.format(0))

        self.lane_detection_network = lane_detection_network(self.p)

        self.setup_optimizer()
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[40, 80, 120, 160, 200, 240, 280, 320, 360, 400], gamma=0.5)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[70, 120, 150, 200], gamma=0.5)

        self.current_epoch = 0

        self.hm_criterion = util.FocalLoss()
        self.lane_criterion = torch.nn.functional.smooth_l1_loss

    def count_parameters(self, model):
	    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_optimizer(self):
        params = []
        weight_decay = self.p.weight_decay
        lr = self.p.l_rate
        for key, value in self.lane_detection_network.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        self.optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay)

       
    #####################################################
    ## train
    #####################################################
    def train(self, inputs, target_lanes, target_h, epoch):
        point_loss, info_dict = self.train_point(inputs, target_lanes, target_h, epoch)
        return point_loss, info_dict

    #####################################################
    ## compute loss function and optimize
    #####################################################
    def train_point(self, inputs, target_lanes, target_h, epoch):
        real_batch_size = len(target_lanes)

        #generate ground truth
        ground_truth_point, meta = util.make_ground_truth_point(target_lanes, target_h, self.p)
        #util.visualize_gt(ground_truth_point[0], meta, inputs[0], self.p)

        # convert numpy array to torch tensor
        ground_truth_point = torch.from_numpy(ground_truth_point).float()
        ground_truth_point = Variable(ground_truth_point).cuda()
        ground_truth_point.requires_grad=False

        # convert numpy array to torch tensor
        gt_heatmaps = meta['end_hm']
        gt_heatmaps = torch.from_numpy(gt_heatmaps).float()
        gt_heatmaps = Variable(gt_heatmaps).cuda()
        gt_heatmaps.requires_grad=False

        # update lane_detection_network
        confidences, htmap, lane_sample, lane_preds = self.predict_lanes(inputs, meta, epoch)
        seg_loss = 0
        for confidance in confidences:

            exist_condidence_loss = 0
            nonexist_confidence_loss = 0

            #segmentation loss
            #exist confidance loss
            confidance_gt = ground_truth_point[:, 0, :, :]
            confidance_gt = confidance_gt.view(real_batch_size, 1, self.p.grid_y, self.p.grid_x)
            exist_condidence_loss = torch.sum(    (confidance_gt[confidance_gt==1] - confidance[confidance_gt==1])**2      )/torch.sum(confidance_gt==1)

            #non exist confidance loss
            nonexist_confidence_loss = torch.sum(    (confidance_gt[confidance_gt==0] - confidance[confidance_gt==0])**2      )/torch.sum(confidance_gt==0)

            seg_loss += self.p.constant_exist*exist_condidence_loss + self.p.constant_nonexist*nonexist_confidence_loss

        heatmap_loss = 0
        heatmap_loss = self.hm_criterion(util.sigmoid(htmap), gt_heatmaps)

        lane_loss = 0
        lane_loss = self.lane_criterion(lane_preds, lane_sample['lane_target'])/ len(lane_sample['lane_target'])

        lane_detection_loss = 0
        #lane_detection_loss = seg_loss + 0.l * heatmap_loss + 10 * lane_loss 
        lane_detection_loss = seg_loss + 0.5 * heatmap_loss + lane_loss 

        info_dict = OrderedDict([('seg_loss', seg_loss.item()),
                                 ('heatmap_loss', heatmap_loss.item()),
                                 ('lane_loss', lane_loss.item()),
                                 ('lane_detection_loss', lane_detection_loss.item()),
                                ])

        self.optimizer.zero_grad()
        lane_detection_loss.backward()
        self.optimizer.step()

        del confidance
        del ground_truth_point
        del exist_condidence_loss, nonexist_confidence_loss, lane_loss

        #if epoch>0 and epoch%80==0 and self.current_epoch != epoch:
        #    self.current_epoch = epoch
        #    #if epoch>0 and (epoch == 1000):
        #    self.p.constant_lane_loss += 0.5
        #    self.p.constant_nonexist += 0.5
        #    self.p.l_rate /= 2.0
        #    self.setup_optimizer()

        return lane_detection_loss, info_dict

    #####################################################
    ## predict lanes
    #####################################################
    def predict_lanes(self, inputs, meta, nepoch):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()

        return self.lane_detection_network(inputs, meta, nepoch)

    #####################################################
    ## predict lanes in test
    #####################################################
    def predict_lanes_test(self, inputs, meta):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()

        return self.lane_detection_network(inputs, meta, self.p.dynamic_sample_test)

    #####################################################
    ## Training mode
    #####################################################                                                
    def training_mode(self):
        self.lane_detection_network.train()

    #####################################################
    ## evaluate(test mode)
    #####################################################                                                
    def evaluate_mode(self):
        self.lane_detection_network.eval()

    #####################################################
    ## Setup GPU computation
    #####################################################                                                
    def cuda(self):
        self.lane_detection_network.cuda()

    #####################################################
    ## Load save file
    #####################################################

    def load_weights(self, model):
        model_path = self.p.save_path+model
        check_point = torch.load(model_path) 
        print('load {}, epoch {}'.format(model, check_point['epoch']))
        self.lane_detection_network.load_state_dict(check_point['state_dict'])
        self.optimizer.load_state_dict(check_point['optimizer'])
        self.scheduler.load_state_dict(check_point['scheduler'])
        start_epoch = check_point['epoch'] + 1
        self.p.l_rate = self.get_lr()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(self.p.device, non_blocking=True)

        return start_epoch 
 

    #####################################################
    ## Save model
    #####################################################
    def save_model(self, epoch, path):
        save_path = self.p.save_path+path
        data = {'epoch': epoch,
                'state_dict': self.lane_detection_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
               }
        #torch.save(data, save_path, _use_new_zipfile_serialization=False)
        torch.save(data, save_path)

    def get_lr(self):
        optimizer = self.optimizer
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


