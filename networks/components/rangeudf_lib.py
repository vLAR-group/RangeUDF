import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.components.pytorch_lib as pt_utils


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = Ops.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        f_neighbours = Ops.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        f_neighbours = Ops.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg


class attentive_interpolation(nn.Module):
    def __init__(self, d_out,task_flag,distance=True):
        super().__init__()
        self.dis=distance
        if task_flag=='rec':
            if distance:
                self.mlp1 = pt_utils.Conv2d(10, d_out, kernel_size=(1, 1), bn=True)
            else:
                self.mlp1 = pt_utils.Conv2d(9, d_out, kernel_size=(1, 1), bn=True)
            self.att_pooling_1 = Att_pooling(2 * d_out, d_out)
        else:
            self.att_pooling_1 = Att_pooling(3+d_out, d_out)
        
        self.task=task_flag

    def forward(self, on_xyz, off_xyz, feature, neigh_idx):
        # print(feature.shape)
        ##############################################
        # relative_pos_encoding using 2 sets of points
        ##############################################
        
        neighbor_xyz = Ops.gather_neighbour(on_xyz, neigh_idx)
        off_xyz_tile = off_xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = off_xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        if self.dis:
            relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        if self.task=='rec':
            if self.dis:
                f_xyz = torch.cat([relative_dis, relative_xyz, off_xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
            else:
                f_xyz = torch.cat([ relative_xyz, off_xyz_tile, neighbor_xyz], dim=-1)   # batch*npoint*nsamples*9
            f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
            f_xyz = self.mlp1(f_xyz)
        else:
            f_xyz = neighbor_xyz.permute((0, 3, 1, 2))  # batch*npoint*nsamples*3
        ##############################################

        f_neighbours = Ops.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        # print(f_neighbours.shape)
        # print(f_xyz.shape)
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        return f_pc_agg


class Ops(nn.Module):

    @staticmethod
    def relative_pos_encoding(xyz, neigh_idx):
        neighbor_xyz = Ops.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = neighbor_idx.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features
