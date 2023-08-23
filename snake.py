import torch.nn as nn
import torch


def conv1d(state_dim, out_state_dim=None, n_adj=4):
    out_state_dim = state_dim if out_state_dim is None else out_state_dim
    return nn.Conv1d(state_dim, out_state_dim, kernel_size=n_adj*2+1)

def dilatedconv1d(state_dim, out_state_dim=None, n_adj=4, dilation=1, padding=4):
    out_state_dim = state_dim if out_state_dim is None else out_state_dim
    return nn.Conv1d(state_dim, out_state_dim, kernel_size=n_adj*2+1, padding=padding, dilation=dilation)

_conv_factory = {
    'grid': conv1d,
    'dgrid': dilatedconv1d
}

class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1, padding=4):
        super(BasicBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation, padding)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class Snake(nn.Module):
    # state_dim = 128, feature_dim = 64 + 2 
    def __init__(self, state_dim, feature_dim, conv_type='dgrid'):
        super(Snake, self).__init__()

        # feature_dim = 64 + 2, state_dim = 128, conv_type='dgrid'
        self.head = BasicBlock(feature_dim, state_dim, conv_type)

        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 5, 5]
        padding  = [4, 4, 4, 8, 8, 20, 20]
        for i in range(self.res_layer_num):
            conv = BasicBlock(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i], padding=padding[i])
            self.__setattr__('res'+str(i), conv)

        fusion_state_dim = 256
        self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x):
        states = []

        # x shape: [num_of_lanes, 66, 56]
        x = self.head(x)
        # x shape after head: [num_of_lanes, 128, 56]

        states.append(x)
        # x shape: [num_of_lanes, 128, 56]
        for i in range(self.res_layer_num):
            x = self.__getattr__('res'+str(i))(x) + x
            states.append(x)

        # state shape: [num_of_lanes, 128x8, 56]
        state = torch.cat(states, dim=1)
        # state shape after fusion: [num_of_lanes, 128x8 -> 256, 56]
        state_tmp = self.fusion(state)
        # [num_of_lanes, 256, 1] 
        global_state = torch.max(state_tmp, dim=2, keepdim=True)[0]
        # [num_of_lanes, 256, 56] 
        global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
        # [num_of_lanes, 1280, 56] 
        state = torch.cat([global_state, state], dim=1)
        # [num_of_lanes, 1, 56] 
        x = self.prediction(state)

        return x

class Evolution(nn.Module):
    def __init__(self, p):
        super(Evolution, self).__init__()
        self.p = p

        #self.evolve_gcn = Snake(state_dim=128, feature_dim=128+2, conv_type='dgrid')
        self.evolve_gcn = Snake(state_dim=128, feature_dim=256+2, conv_type='dgrid')
        #self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
        for i in range(self.p.iters):
            #evolve_gcn = Snake(state_dim=128, feature_dim=128+2, conv_type='dgrid')
            #evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
            evolve_gcn = Snake(state_dim=128, feature_dim=256+2, conv_type='dgrid')
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)
            
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_gcn_feature(self, cnn_feature, img_poly, ind, h, w):
        img_poly = img_poly.clone()
        img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
        img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
    
        batch_size = cnn_feature.size(0)
        gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
        for i in range(batch_size):
            poly = img_poly[ind == i].unsqueeze(0)
            #feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly, align_corners=True)
            feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)
            feature = feature[0].permute(1, 0, 2)
            gcn_feature[ind == i] = feature
    
        return gcn_feature

    def forward(self, lane_sample, cnn_feature):
        n_batch, _, h, w = cnn_feature.shape

        lane_ind = lane_sample['lane_ind']
        lane_init = lane_sample['lane_init']
        init_feature = self.get_gcn_feature(cnn_feature, lane_init, lane_ind, h, w)
        init_input = torch.cat([init_feature, lane_init.permute(0, 2, 1)], dim=1)
        tmp_lane = self.evolve_gcn(init_input)
        lane_pre = tmp_lane.permute(0, 2, 1) + lane_init
        
        for i in range(self.p.iters):
            lane_init = lane_pre
            init_feature = self.get_gcn_feature(cnn_feature, lane_init, lane_ind, h, w)
            init_input = torch.cat([init_feature, lane_pre.permute(0, 2, 1)], dim=1)
            evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
            tmp_lane = evolve_gcn(init_input)
            lane_pre = tmp_lane.permute(0, 2, 1) + lane_init 

        return lane_pre

