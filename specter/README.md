# SPECTER

Signal intelligence ML system for covert radio transmission detection using RadioML 2018.01A.

## Setup

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Dataset

Download RadioML 2018.01A HDF5 from https://www.deepsig.ai/datasets and place at `data/GOLD_XYZ_OSC.0001_1024.hdf5`.

## Usage

```bash
python train.py
python evaluate.py
```
