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


from shapely.geometry import Polygon
from osgeo import gdal, gdal_array
gdal.UseExceptions()
import laspy
from laxpy.tree import LAXTree
from laxpy.file import LAXParser


from pn2_scalar_regressor import Net
import rasterizer

class PointCloudsInFiles(InMemoryDataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(self, lasfiles, biomassfile, max_points=20000):
        self.lasfiles = lasfiles
        self.biomassfile = biomassfile
        self.max_points = max_points

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
        self.valid_1d_indices = np.where(np.isfinite(arr.flatten()))[0]
        self.biomass = arr



        self.bboxes = []
        for lasfile in self.lasfiles:
            laxfile = re.sub(r'^(.*).la[sz]$', r'\1.lax', str(lasfile))
            parser = LAXParser(laxfile)
            tree = LAXTree(parser)
            minx, maxx, miny, maxy = parser.bbox
            bbox_las = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
            self.bboxes.append(bbox_las)

        super().__init__()

    def __len__(self):
        return len(self.valid_1d_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xpos, ypos = np.unravel_index(self.valid_1d_indices[idx], (self.YSize, self.XSize))
        y_label = self.biomass[xpos, ypos]
        bbox = [self.ulx + (ypos+0.0) * self.xres,
                self.uly + (xpos+0.0) * self.yres,
                self.ulx + (ypos+1.0) * self.xres,
                self.uly + (xpos+1.0) * self.yres]

        print(bbox, y_label)
        q_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])

        pcloud = {'xyz': []}
        for lasfile_id, lasfile in enumerate(self.lasfiles):
            bbox_las = self.bboxes[lasfile_id]
            if not q_polygon.intersects(bbox_las):
                continue

            laxfile = re.sub(r'^(.*).la[sz]$', r'\1.lax', str(lasfile))
            parser = LAXParser(laxfile)
            tree = LAXTree(parser)

            lashandle = laspy.read(lasfile)

            # x, y = lashandle.x, lashandle.y
            # valid_points = np.logical_and(
            #     np.logical_and(x > bbox[0], x <= bbox[2]),
            #     np.logical_and(y > bbox[3], y <= bbox[1]),
            # )
            # # valid_indices = np.where(valid_points)
            # pcloud['xyz'].append(lashandle.xyz[valid_points, :])


            for cell_index, polygon in tree.cell_polygons.items():  # use quadtree for preselection
                if q_polygon.intersects(polygon):
                    candidate_indices = parser.create_point_indices(cell_index)
                    x, y = lashandle.x[candidate_indices], lashandle.y[candidate_indices]
                    valid_points = np.logical_and(
                        np.logical_and(x > bbox[0], x <= bbox[2]),
                        np.logical_and(y > bbox[3], y <= bbox[1]),
                    )
                    valid_indices = candidate_indices[valid_points]
                    pcloud['xyz'].append(lashandle.xyz[valid_indices, :])
        if len(pcloud['xyz']) > 0:
            pcloud['xyz'] = np.concatenate(pcloud['xyz'], axis=0)
        if pcloud['xyz'].shape[0] < self.max_points:
            pcloud['xyz'] = pcloud['xyz'][np.random.choice(range(pcloud['xyz'].shape[0]), self.max_points, replace=True), :]
        elif pcloud['xyz'].shape[0] > self.max_points:
            pcloud['xyz'] = pcloud['xyz'][np.random.choice(range(pcloud['xyz'].shape[0]), self.max_points, replace=False), :]

        sample = Data(pos=torch.from_numpy(pcloud['xyz']).float(),
                      y=torch.as_tensor(y_label, dtype=torch.float))

        return sample

class PointCloudsRasterExtract(InMemoryDataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(self, lasfiles, biomassfile, max_points=20000, backup_extract=None):
        self.lasfiles = lasfiles
        self.biomassfile = biomassfile
        self.max_points = max_points

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



        self.bboxes = []
        self.lasindices = np.full_like(self.biomass, None, dtype=object)

        if backup_extract is not None and os.path.exists(backup_extract):
            print("Loading previously extracted point indices from disk...")
            self.lasindices = np.load(backup_extract, allow_pickle=True)

        else:
            for fileid, lasfile in enumerate(tqdm.tqdm(self.lasfiles, "scanning input las files")):
                lashandle = laspy.read(lasfile)
                xy = lashandle.xyz[:, :2]
                raster = rasterizer.Rasterizer(data=xy, raster_size=(xres, yres), method=None)
                XVoxelCenter, XVoxelContains, idxVoxelUnique, _ = raster.rasterize(origin=(ulx, uly))

                for cid, (centerx, centery, contains) in enumerate(zip(*XVoxelCenter, XVoxelContains)):
                    # get respective biomass
                    px = int((centerx - ulx)/xres)
                    py = int((centery - uly)/yres)
                    if px < 0 or px >= self.XSize or py < 0 or py >= self.YSize:
                        continue
                    bm = arr[py, px]
                    if not np.isnan(bm):
                        if self.lasindices[py, px] is None:
                            self.lasindices[py, px] = {fileid: contains}
                        else:
                            self.lasindices[py, px][fileid] = contains
            if backup_extract is not None:
                self.lasindices.dump(backup_extract)

        self.valid_1d_indices = np.where(np.logical_and(np.isfinite(arr.flatten()), self.lasindices.flatten()))[0]

        super().__init__()

    def __len__(self):
        return len(self.valid_1d_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xpos, ypos = np.unravel_index(self.valid_1d_indices[idx], (self.YSize, self.XSize))
        y_label = self.biomass[xpos, ypos]
        lasindices = self.lasindices[xpos, ypos]

        xyz = []
        for fileid, contains in lasindices.items():
            lashandle = laspy.read(self.lasfiles[fileid])
            xyz.append(lashandle.xyz[contains])
        xyz = np.concatenate(xyz)
        sample = Data(pos=torch.from_numpy(xyz).float(),
                      y=torch.as_tensor(y_label, dtype=torch.float))

        return sample



def main(args):
    train_dataset = PointCloudsRasterExtract(lasfiles=list(Path(r"D:\lwiniwar\data\uncertaintree\PetawawaHarmonized\Harmonized\2012_ALS\3_tiled_norm").glob("*.laz")),
                               biomassfile=r"D:\lwiniwar\data\uncertaintree\DeepBiomass\RF_PRF_biomass_Ton_DRY_masked_train.tif",
                                             backup_extract=r"D:\lwiniwar\data\uncertaintree\DeepBiomass\train_presel.npy"
                               )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              num_workers=6)

    test_dataset = PointCloudsRasterExtract(lasfiles=list(Path(r"D:\lwiniwar\data\uncertaintree\PetawawaHarmonized\Harmonized\2012_ALS\3_tiled_norm").glob("*.laz")),
                               biomassfile=r"D:\lwiniwar\data\uncertaintree\DeepBiomass\RF_PRF_biomass_Ton_DRY_masked_test.tif",
                                             backup_extract=r"D:\lwiniwar\data\uncertaintree\DeepBiomass\test_presel.npy"
                               )
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
                              num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device.")
    model = Net(num_features=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


    def train():
        model.train()
        loss_list = []
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)[:, 0]
            loss = F.mse_loss(out, data.y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print(f'[{i + 1}/{len(train_loader)}] MSE Loss: {loss.to("cpu"):.4f} ')
                loss_list.append(loss.detach().to("cpu").numpy())
        print(f'mean loss this epoch: {np.mean(loss_list)}')
        return np.mean(loss_list)

    @torch.no_grad()
    def test(loader, ep_id):
        model.eval()
        losses = []
        for idx, data in enumerate(loader):
            data = data.to(device)
            outs = model(data)
            loss = F.mse_loss(outs, data.y)
            print("Sample differences in p95:")
            print(data.y.to('cpu').numpy() - outs.to('cpu').numpy())
            losses.append(float(loss.to("cpu")))
        return float(np.mean(losses))


    for epoch in range(1, 1001):
        model_path = rf'D:\lwiniwar\data\uncertaintree\DeepBiomass\models\deepbiomass.model'
        if os.path.exists(model_path):
            model = torch.load(model_path)
        train_mse = train()

        mse = test(test_loader, epoch)
        torch.save(model, model_path)
        with open(model_path.replace('.model', '.csv'), 'a') as f:
            f.write(
            f'{epoch}, {train_mse}, {mse}\n'
            )
        print(f'Epoch: {epoch:02d}, Mean test MSE: {mse:.4f}')
        print(f'Epoch: {epoch:02d}, Mean train MSE: {train_mse:.4f}')


if __name__ == '__main__':
    main(sys.argv[1:])