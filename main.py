import sys
from pathlib import Path
import re
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import Sampler

from shapely.geometry import Polygon
from osgeo import gdal, gdal_array
gdal.UseExceptions()
import laspy
from laxpy.tree import LAXTree
from laxpy.file import LAXParser


from pn2_scalar_regressor import Net
import rasterizer
from HDF5Loader import HDF5BiomassPointCloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, scheduler, train_loader, device, path):
    model.train()
    loss_list = []
    for i, data in enumerate(tqdm.tqdm(train_loader, desc="Training")):
        if data.y.shape[-1] != train_loader.batch_size:
            print("Skipping last batch (not a full batch")
            continue
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)[:, 0]
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 1 == 0:
            l = loss.detach().to("cpu").numpy()
            # print(f'[{i + 1}/{len(train_loader)}] RMSE Loss: {np.sqrt(l):.4f} ')
            loss_list.append(l)
        if (i + 1) % 1000 == 0:
            print(f'mean RMSE loss last 1000 it: {np.mean(np.sqrt(loss_list[-1000:]))}')

    scheduler.step()
    print(f'mean RMSE loss this epoch: {np.sqrt(np.mean(loss_list))}')
    print(f'mean RMSE loss last 1000 it: {np.sqrt(np.mean(loss_list[-1000:]))}')

    return np.mean(loss_list)


@torch.no_grad()
def test(model, loader):
    model.eval()
    losses = []
    for idx, data in enumerate(tqdm.tqdm(loader, desc="Testing")):
        data = data.to(device)
        outs = model(data)[:, 0]
        loss = F.mse_loss(outs, data.y)

        if (idx + 1) % 1000 == 0:
            print("Sample differences in biomass:")
            print(data.y.to('cpu').numpy(), ' - ', outs.to('cpu').numpy(), ' = ',
                  data.y.to('cpu').numpy() - outs.to('cpu').numpy())
        losses.append(float(loss.to("cpu")))
    return float(np.mean(losses))


def main(args):
    lr = float(args[0])
    min_lr = float(args[1])
    n_points = int(args[2])
    bs = int(args[3])
    
    train_dataset = HDF5BiomassPointCloud(lasfiles=list(Path(r"/tmp/lwiniwar/2012_norm").glob("*.laz")),
                               biomassfile=os.path.expandvars(r"$DATA/PetawawaHarmonized/RF_PRF_biomass_Ton_DRY_masked_train.tif"),
                                             backup_extract=os.path.expandvars(r"/tmp/lwiniwar/2012_norm/train_presel.hdf5"),
                                             max_points=n_points
                               )
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=False,
                              num_workers=16)

    test_dataset = HDF5BiomassPointCloud(lasfiles=list(Path(r"/tmp/lwiniwar/2012_norm").glob("*.laz")),
                               biomassfile=os.path.expandvars(r"$DATA/PetawawaHarmonized/RF_PRF_biomass_Ton_DRY_masked_val.tif"),
                                             backup_extract=os.path.expandvars(r"/tmp/lwiniwar/2012_norm/val_presel.hdf5"),
                                             max_points=n_points
                               )
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                              num_workers=16)

    print(f"Using {device} device.")
    model = Net(num_features=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=dc)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=min_lr, last_epoch=-1, verbose=True)
    model_path = os.path.expandvars(
        rf'$DATA/PetawawaHarmonized/models/deepbiomass_lr{lr}_minLR{min_lr}_bs{bs}_{n_points}points_3PN_noload.model')

    if os.path.exists(model_path):
        model = torch.load(model_path)
        print('loading existing model')

    for epoch in range(1, 1001):
        train_mse = train(model, optimizer, scheduler, train_loader, device, model_path)
        torch.save(model, model_path)
        test_mse = test(model, test_loader)

        with open(model_path.replace('.model', '.csv'), 'a') as f:
            f.write(
            f'{epoch}, {train_mse}, {test_mse}, {optimizer.param_groups[0]["lr"]}\n'
            )
        print(f'Epoch: {epoch:02d}, Mean test MSE: {test_mse:.4f}')
        print(f'Epoch: {epoch:02d}, Mean train MSE: {train_mse:.4f}')


if __name__ == '__main__':
    print('lr min_lr num_points batch_size')
    main(sys.argv[1:])
