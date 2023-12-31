import torch
from torch_geometric.nn import MLP, knn_interpolate, PointNetConv, global_max_pool, fps, radius

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.1, MLP([3 + num_features, 64, 64, 128], act='LeakyReLU'))
        self.sa2_module = SAModule(0.25, 0.5, MLP([128 + 3, 128, 128, 256], act='LeakyReLU'))
        #self.sa3_module = SAModule(0.25, 0.5, MLP([256 + 3, 256, 512, 512], act='LeakyReLU'))
        # self.sa4_module = GlobalSAModule(MLP([512 + 3, 1024, 1024, 2048], act='LeakyReLU'))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024], act='LeakyReLU'))
        # self.sa3_module = GlobalSAModule(MLP([128 + 3, 256, 512, 1024], act='LeakyReLU'))

        # self.mlp = MLP([2048, 256, 256, 1], act='LeakyReLU')
        self.mlp = MLP([1024, 256, 256, 1], act='LeakyReLU')

    def forward(self, data, return_feature_vec=False):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        #sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = self.sa3_module(*sa2_out)
        if return_feature_vec:
            return self.mlp(x), x
        else:
            return self.mlp(x)
