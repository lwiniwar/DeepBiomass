import sys, os
import tqdm

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP
from torch_geometric.loader import DataLoader

from pn2_scalar_regressor import Net as PNet2
from HDF5Loader import TwoHDF5BiomassPointCloud



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.PN1 = PNet2(0)
#         self.mlp = MLP([1024, 2048, 2048, 2048, 1024], act='LeakyReLU')
#
#     def forward(self, data):
#
#         t1_sa0_out = (data.x1, data.pos1, data.batch)
#         t1_sa1_out = self.PN1.sa1_module(*t1_sa0_out)
#         t1_sa2_out = self.PN1.sa2_module(*t1_sa1_out)
#         t1_x, _, _ = self.PN1.sa3_module(*t1_sa2_out)
#
#         t2_sa0_out = (data.x2, data.pos2, data.batch)
#         t2_sa1_out = self.PN1.sa1_module(*t2_sa0_out)
#         t2_sa2_out = self.PN1.sa2_module(*t2_sa1_out)
#         t2_x, _, _ = self.PN1.sa3_module(*t2_sa2_out)
#
#         return self.mlp(t1_x), t2_x

class JustMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP([1024, 2048, 2048, 2048, 1024], act='LeakyReLU')
    def forward(self, data):
        return self.mlp(data)

def train(biomass_model, predict_model, optimizer, dataloader):
    predict_model.train()
    loss_list = []
    for i, data in enumerate(tqdm.tqdm(dataloader, desc="Training")):
        if len(data.t1) != dataloader.batch_size:
            print("Skipping last batch (not a full batch")
            continue
        data_t1 = data.t1.to(device)
        data_t2 = data.t2.to(device)
        optimizer.zero_grad()

        # get the feature vector for both epochs from the previously trained NN
        biomass_t1, features_t1 = biomass_model(data_t1, return_feature_vec=True)
        biomass_t2, features_t2 = biomass_model(data_t2, return_feature_vec=True)  # note that this biomass is not valid

        features_t2_from_t1 = predict_model(features_t1)
        loss = F.mse_loss(features_t2, features_t2_from_t1)  # minimize difference in feature vector space (1024-dim)

        loss.backward()
        optimizer.step()

def test(biomass_model, predict_model, dataloader):
    predict_model.eval()
    loss_list = []
    for i, data in enumerate(tqdm.tqdm(dataloader, desc="Testing")):
        if data.y.shape[-1] != dataloader.batch_size:
            print("Skipping last batch (not a full batch")
            continue
        data = data.to(device)
        biomass_t1, features_t1 = biomass_model(data.t1, return_feature_vec=True)
        biomass_t2, features_t2 = biomass_model(data.t2, return_feature_vec=True)

        features_t2_from_t1 = predict_model(features_t1)
        biomass_t2_from_t1 = biomass_model.mlp(features_t2_from_t1)
        diff_biomass = biomass_t2 - biomass_t2_from_t1
        loss_list.append(diff_biomass)

def main(args):
    lr = float(args[0] if len(args)>0 else 1e-5)
    min_lr = float(args[1] if len(args)>1 else 1e-8)

    trained_biomass_model = PNet2(0)

    trained_biomass_model_w = torch.load(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\models\deepbiomass_lr3e-06_minLR1e-08_bs8_8192pts_normXYZ.model", map_location=device)
    trained_biomass_model.load_state_dict(trained_biomass_model_w.state_dict())


    train_dataset = TwoHDF5BiomassPointCloud(
        lasfiles_ep1=list(Path(r"D:\lwiniwar\data\uncertaintree\PetawawaHarmonized\Harmonized\2012_ALS\3_tiled_norm").glob("*.laz")),
        lasfiles_ep2=list(Path(r"D:\lwiniwar\data\uncertaintree\PetawawaHarmonized\Harmonized\2018_SPL_CGVD28\3_tiled_norm").glob("*.laz")),
        biomassfile=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\RF_PRF_biomass_Ton_DRY_masked_train.tif"),
        backup_extract_ep1=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\train_presel.hdf5"),
        backup_extract_ep2=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\train_presel_2018.hdf5"),
        max_points=2048)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=6)

    test_dataset = TwoHDF5BiomassPointCloud(
        lasfiles_ep1=list(
            Path(r"D:\lwiniwar\data\uncertaintree\PetawawaHarmonized\Harmonized\2012_ALS\3_tiled_norm").glob("*.laz")),
        lasfiles_ep2=list(
            Path(r"D:\lwiniwar\data\uncertaintree\PetawawaHarmonized\Harmonized\2018_SPL_CGVD28\3_tiled_norm").glob(
                "*.laz")),
        biomassfile=os.path.expandvars(
            r"D:\lwiniwar\data\uncertaintree\DeepBiomass\RF_PRF_biomass_Ton_DRY_masked_val.tif"),
        backup_extract_ep1=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\val_presel.hdf5"),
        backup_extract_ep2=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\val_presel_2018.hdf5"),
        max_points=2048)

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                              num_workers=6)


    predict_model = JustMLP()
    optimizer = torch.optim.Adam(predict_model.parameters(), lr=lr)  # , weight_decay=dc)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=min_lr, last_epoch=-1,
                                                           verbose=True)

    for ep in range(50):
        train(trained_biomass_model, predict_model, optimizer, train_loader)
        scheduler.step()
        test(trained_biomass_model, predict_model, test_loader)



if __name__ == '__main__':
    main(sys.argv[1:])