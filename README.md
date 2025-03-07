# S³AD: Semi-supervised Small Apple Detection in Orchard Environments

## Usage

### Requirements
- `Debian 12.2.0-14`
- `Anaconda3` with `python=3.8`
- `Pytorch=1.13.1`
- `mmdetection=2.26.0+`
- `mmcv=1.7.1`


### Installation
```
make install
```

### Installation

- `Replace in the newly installed mmdetection repo the following lines of code:`
- `thirdparty/mmdetection/mmdet/datasets/coco.py (adjust CLASSES and PALETTE to new dataset)`
  - `CLASSES = ('apple',)`
  - `PALETTE = [(220, 20, 60),] `
- `thirdparty/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py`
  -  `ln 47: num_classes=1 `
- `thirdparty/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py`
  -  `ln 109: score_th = 0.001`

### Data Preparation
  - Download the [MAD]([https://wandb.ai/](https://www2.informatik.uni-hamburg.de/cv/projects/MAD.zip)) dataset
- `  `
- `  `
  
