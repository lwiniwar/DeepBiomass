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

import h5py

from osgeo import gdal
import laspy
import rasterizer

class HDF5BiomassPointCloud(InMemoryDataset):
    """Point cloud dataset."""

    def __init__(self, lasfiles, biomassfile, max_points=20000, backup_extract=None, skip_n=0, red_point=None):
        self.lasfiles_ep1 = lasfiles
        self.biomassfile = biomassfile
        self.max_points = max_points
        self.skip_n = skip_n
        self.backup_extract = backup_extract

        ds = gdal.Open(self.biomassfile)
        ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
        lrx = ulx + (ds.RasterXSize * xres)
        lry = uly + (ds.RasterYSize * yres)

        self.XSize = ds.RasterXSize
        self.YSize = ds.RasterYSize
        self.ulx = ulx
        self.uly = uly
        self.xres = xres
        self.yres = yres

        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        arr = band.ReadAsArray()
        arr[arr == nodata] = np.nan
        self.biomass = arr
        valid_loc = np.full_like(arr, False, dtype=bool)

        if self.backup_extract is not None and os.path.exists(self.backup_extract):
            print("Extracted keys already exist...")
            with h5py.File(self.backup_extract, "r") as f:
                valid_loc = f["valid_loc"][...]

        else:
            f = h5py.File(self.backup_extract, "w")

            for fileid, lasfile in enumerate(tqdm.tqdm(self.lasfiles_ep1, "scanning input las files")):
                lashandle = laspy.read(lasfile)
                xyz = lashandle.xyz
                xy = xyz[:, :2]
                minxy = np.min(xy, axis=0)
                maxxy = np.max(xy, axis=0)
                if red_point is None:
                    red_point = np.array([minxy[0], minxy[0], 0])
                if maxxy[0] < ulx or minxy[0] > lrx:
                    continue
                if maxxy[1] < lry or minxy[1] > uly:
                    continue
                raster = rasterizer.Rasterizer(data=xy, raster_size=(xres, yres), method=None)
                XVoxelCenter, XVoxelContains, idxVoxelUnique, _ = raster.rasterize(origin=(ulx, uly))
                for cid, (centerx, centery, contains) in enumerate(zip(*XVoxelCenter, XVoxelContains)):
                    # get respective biomass
                    px = int((centerx - ulx - xres/2)/xres)
                    py = int((centery - uly - yres/2)/yres)
                    if px < 0 or px >= self.XSize or py < 0 or py >= self.YSize:
                        continue
                    bm = arr[py, px]
                    if not np.isnan(bm):
                        valid_loc[py, px] = True
                        key_name = f'{py}_{px}'
                        if key_name not in f.keys():
                            g = f.create_group(key_name)
                        else:
                            g = f[key_name]
                        dset = g.create_dataset(f"{fileid}", (len(contains),3), float, compression="lzf")
                        dset[...] = xyz[contains] - red_point
            dset = f.create_dataset("valid_loc", valid_loc.shape, valid_loc.dtype, compression="lzf")
            dset[...] = valid_loc
            f.close()

        self.valid_1d_indices = np.where(valid_loc.flatten())[0]
        self.red_point = red_point
        print(f"{len(self)} training samples found.")
        super().__init__()

    def __len__(self):
        return len(self.valid_1d_indices) - self.skip_n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx += self.skip_n

        xpos, ypos = np.unravel_index(self.valid_1d_indices[idx], (self.YSize, self.XSize))
        y_label = self.biomass[xpos, ypos]

        with h5py.File(self.backup_extract, 'r') as f:
            points = []
            g = f[f"{xpos}_{ypos}"]
            for key in g.keys():
                points.append(g[key][...])

        xyz = np.concatenate(points, axis=0)

        xyz = xyz[np.random.choice(range(xyz.shape[0]), self.max_points,
                                   replace=xyz.shape[0] < self.max_points), :]

        xyz -= np.mean(xyz, axis=0)
        # print(xyz.shape[0])
        # for fileid, contains in lasindices.items():
        #     lashandle = laspy.read(self.lasfiles[fileid])
        #     xyz.append(lashandle.xyz[contains])
        # xyz = np.concatenate(xyz)
        sample = Data(pos=torch.from_numpy(xyz).float(),
                      y=torch.as_tensor(y_label, dtype=torch.float))

        return sample

if __name__ == '__main__':
    train_dataset = HDF5BiomassPointCloud(lasfiles=list(Path(r"D:\lwiniwar\data\uncertaintree\PetawawaHarmonized\Harmonized\2012_ALS\3_tiled_norm").glob("*.laz")),
                               biomassfile=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\RF_PRF_biomass_Ton_DRY_masked_train.tif"),
                                             backup_extract=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\train_presel.hdf5"),
                                             max_points=2048
                               )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False,
                              num_workers=6)

    test_dataset = HDF5BiomassPointCloud(lasfiles=list(Path(r"D:\lwiniwar\data\uncertaintree\PetawawaHarmonized\Harmonized\2012_ALS\3_tiled_norm").glob("*.laz")),
                               biomassfile=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\RF_PRF_biomass_Ton_DRY_masked_val.tif"),
                                             backup_extract=os.path.expandvars(r"D:\lwiniwar\data\uncertaintree\DeepBiomass\val_presel.hdf5"),
                                             max_points=8096
                               )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False,
                              num_workers=6)

    sum = 0
    count = 0
    for data in tqdm.tqdm(test_loader, "Test loading..."):
        sum += data.pos.shape[0]
        count += data.y.shape[-1]
    print(sum/count)

    for data in tqdm.tqdm(train_loader, "Train loading..."):
        pass