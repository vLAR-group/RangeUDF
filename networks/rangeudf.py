import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.components.pytorch_lib as pt_utils
from networks.components.rangeudf_lib import Dilated_res_block, Ops, attentive_interpolation

class BackBone(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("initializing backbone")
        self.config = config
        self.fc0 = pt_utils.Conv1d(self.config.in_dim, 8, kernel_size=1, bn=True)
        #encoder
        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        #decoder
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < self.config.num_layers -1:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-self.config.num_layers]
                d_out = 2 * self.config.d_out[-self.config.num_layers]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))
    def forward(self,coords, feature, neigh_idx, sub_idx, on_interp_idx):
        feature = feature.transpose(-2, -1)
        features = self.fc0(feature) # Batch*channel*npoints
        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, coords[i], neigh_idx[i])

            f_sampled_i = Ops.random_sample(f_encoder_i, sub_idx[i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features_on = self.decoder_0(f_encoder_list.pop()) 

        # ###########################Decoder############################
        for j in range(self.config.num_layers):
            f_interp_on = Ops.nearest_interpolation(features_on, on_interp_idx[-j - 1])
            f_on_skip = torch.cat([f_encoder_list.pop(), f_interp_on], dim=1)  # concat on channel dimension

            f_decoder_on = self.decoder_blocks[j](f_on_skip)
            features_on = f_decoder_on
        # ###########################Decoder############################

        return feature, features_on, coords#extract the point-wise feature.

class Task_Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("initializing task_head")
        self.config = config
        self.distance=self.config.distance
        d_out=config.d_out[0]
        d_out=2*d_out
        if 'sem' in self.config.task:
            sem_in_dim=d_out
            sem_out_dim=self.config.num_classes
            self.segmentation_regressor = self.__make_layer(sem_in_dim,self.config.sem_hidden_layers,self.config.sem_hidden_dims,sem_out_dim,task_flag='sem',activation=None)
            self.sem_att_interpolation = attentive_interpolation(d_out,task_flag='sem',distance=self.distance)
        if 'rec' in self.config.task:
            rec_in_dim=d_out + self.config.concat
            rec_out_dim=1
            self.reconstruction_regressor = self.__make_layer(rec_in_dim,self.config.rec_hidden_layers,self.config.rec_hidden_dims,rec_out_dim,task_flag='rec',activation=nn.ReLU())
            self.rec_att_interpolation = attentive_interpolation(d_out,task_flag='rec',distance=self.distance)
            
    def __make_layer(self,input_dim,hidden_layers,hidden_dims,out_dim,task_flag,activation=None):
        layers=[]
        in_channel=input_dim
        for i in range(hidden_layers):
            out_channel=hidden_dims[i]
            layers.append(pt_utils.Conv2d(in_channel,out_channel,kernel_size=(1,1),bn=True))
            in_channel=out_channel
        if task_flag=='sem':
            layers.append(nn.Dropout(0.5))
        head=pt_utils.Conv2d(in_channel,out_dim,kernel_size=(1,1),bn=False,activation=activation)
        layers.append(head)
        return nn.Sequential(*layers)
        
    def forward_rec(self, off_interp_idx, off_coords,feature, features_on, coords):
        # cat point feature
        features_off = self.rec_att_interpolation(coords[0], off_coords, features_on, off_interp_idx[0])
        if self.config.concat == 3:
            features_off = torch.cat((features_off, off_coords.transpose(-2, -1).unsqueeze(dim=3)), dim=1) # d = d_out + 3 = 35
            features_on = torch.cat((features_on, coords[0].transpose(1, 2).unsqueeze(dim=3)), dim=1) # d = d_out + 3 = 35
            features_cat = torch.cat([features_on, features_off], dim=2)  # concat on points dimension
            df_pred = self.reconstruction_regressor(features_cat)   # [B, 1, N, 1]

        elif self.config.concat == 0:
            features_raw = torch.cat([features_on, features_off], dim=2)
            df_pred = self.reconstruction_regressor(features_raw)#distance field

        return df_pred.squeeze(1).squeeze(-1)

    def forward_sem(self, off_interp_idx, off_coords,feature, features_on, coords):
        # cat point feature
        features_off=self.sem_att_interpolation(coords[0], off_coords, features_on, off_interp_idx[0])
        features_raw = torch.cat([features_on, features_off], dim=2)# no coordinates needed for segmentation
        sem_pred = self.segmentation_regressor(features_raw)
        return sem_pred.squeeze(-1)

    def forward(self, off_interp_idx, off_coords,feature, features_on, coords):
        df_pred=None
        sem_pred=None
        if 'rec' in self.config.task:
            df_pred=self.forward_rec(off_interp_idx, off_coords,feature, features_on, coords)
        if 'sem' in self.config.task:
            sem_pred=self.forward_sem(off_interp_idx, off_coords,feature, features_on, coords)
        return df_pred,sem_pred





class D3F(nn.Module):
    def __init__(self, opt):
        super(D3F, self).__init__()
        self.opt = opt
        # print(opt.task)
        self.backbone=BackBone(opt)
        self.net=opt.task
        self.task=Task_Head(opt)

    def forward(self, batch):

        feature = batch.get('feature')
        off_surface_points = batch.get('off_surface_points')
        on_surface_points = [item.to(self.opt.device, non_blocking=True) for item in batch.get('on_surface_points')]
        neigh_idx = [item.to(self.opt.device, non_blocking=True) for item in batch.get('input_neighbors')]
        sub_idx = [item.to(self.opt.device, non_blocking=True) for item in batch.get('input_pools')]
        on_interp_idx = [item.to(self.opt.device, non_blocking=True) for item in batch.get('on_interp_idx')]
        off_interp_idx = [item.to(self.opt.device, non_blocking=True) for item in batch.get('off_interp_idx')]
        feature, features_on, coords=self.backbone(on_surface_points, feature, neigh_idx, sub_idx, on_interp_idx)
        
        df_pred,sem_pred=self.task(off_interp_idx, off_surface_points,feature, features_on, coords)
        if 'joint' in self.net :
            return df_pred,sem_pred
        elif 'rec' in self.net:
            return df_pred
        elif 'sem' in self.net:
            return sem_pred
        else:
            raise ValueError

