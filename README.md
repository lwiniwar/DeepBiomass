# DeepBiomass

This repository holds code to predict biomass (or any raster value) from 3D point cloud data. In a second step, the forecasting of one epoch from a previous one is implemented, based on a feature vector extracted in the course of the biomass prediction.

Data is preprocessed and binned into the pixel locations, and a temporary .hdf5 file is written for fast random read access. To run, change the paths in `main.py` and `main_stage2.py` and execute.

## Requirements

The following python packages are required (conda environment export):
```
name: pytorch1
channels:
  - conda-forge
  - defaults
dependencies:
  - python
  - mamba
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - ca-certificates
  - certifi
  - openssl
  - gdal
  - scikit-image
  - h5py
  - pandas

```

## Funding
This research has been funded in full by the Austrian Research Fund [(FWF)](https://www.fwf.ac.at/de/) \[J 4672-N\].

Development was further supported by the Vienna Scientific Cluster [(VSC5)](https://vsc.ac.at/home/).
