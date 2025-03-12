# S³AD: Semi-supervised Small Apple Detection in Orchard Environments



### Requirements
- `Debian 12.2.0-14`
- `Anaconda3` with `python=3.8`
- `Pytorch=1.13.1`
- `mmdetection=2.26.0+`
- `mmcv=1.7.1`


## Installation

```
make install
```

- Replace in the newly installed mmdetection repo the following lines of code:
   - `thirdparty/mmdetection/mmdet/datasets/coco.py (adjust CLASSES and PALETTE to new dataset)`
     - `CLASSES = ('apple',)`
     - `PALETTE = [(220, 20, 60),] `
   - `thirdparty/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py`
     -  `ln 47: num_classes=1 `
   - `thirdparty/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py`
     -  `ln 109: score_th = 0.001`
- This repo is based on [SoftTeacher](https://github.com/microsoft/SoftTeacher)

## Data Preparation
  - Download the [MAD](https://www2.informatik.uni-hamburg.de/cv/projects/MAD.zip) dataset
  ```
mkdir data
mv MAD.zip data/
cd  data
unzip MAD.zip
rm MAD.zip
  ```
  - Generate the GT for the TreeAttention module
  ```
bash gen_TreeAttentionGt.sh
```
  - Generate Data for SoftTeacher module
  ```
bash tileImgs.sh
  ```

## Training
- TreeAttention
  
  ```
  python training_TreeAttention.py --data_dir "./data/MAD" --save_checkpoint_to "./checkpoints/TreeAttention.h5"
  ```
- SoftTeacher
  - Our model is initialized on the [SoftTeacher COCO weights](https://cloud.uni-hamburg.de/s/gWPKLkiyrAW77aS/download/COCO_softteacher.pth)
  ```
  bash tools/dist_train.sh \
  configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k.py \
  1 \
  --load-from ./checkpoints/COCO_softteacher.pth \
  --lr=0.001
  ```
  
## Testing and Evaluation

- TreeAttention
  
  - Testing / Inference:
     - Our TreeAttention checkpoint can be downloaded [here](https://cloud.uni-hamburg.de/s/DEzmW2ARN7RPmgT/download/MAD_TreeAttention.h5)
    
  ```
  python test_TreeAttention.py \
  --load_imgs_from "./data/MAD/test" \
  --checkpoint "./checkpoints/MAD_TreeAttention.h5" \
  --result_dir "./results/TreeAttention/test/"
  ```
  
  - Evaluation:

  ```
  python evalTreeAttention.py \
  --annotations "./data/MAD/annotations/instances_test.json" \
  --gt "./data/MAD/test_masks" \
  --predictions "./results/TreeAttention/test"
  ```
  
- SoftTeacher
  
  - Testing / Inference:
     - Our SoftTeacher checkpoint can be downloaded [here](https://cloud.uni-hamburg.de/s/iSccQgS79CRPfKW/download/MAD_softteacher.pth)
    
  ```
  python test_SoftTeacher.py \
  --config "./configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k.py" \
  --checkpoint "./checkpoints/MAD_softteacher.pth" \
  --batch 30 \
  --load_imgs_from "./data/MAD/test" \
  --load_tiles_from "./data/MAD_tiles/test" \
  --load_masks_from "./results/TreeAttention/test" \
  --load_anns_from "./data/MAD_tiles/annotations/instances_test.json" \
  --save_results_to "./results/results_softteacher.json" \
  --vis False 
  ```
  
  - Evaluation:

  ```
  python evalCOCO.py --dataset 'data/MAD/annotations/instances_test.json' --results 'results/results_softteacher.json'
  ```

## Citation

  ```bib
  @inproceedings{JohansonEtAlWACV2024,
title = {{S³AD}: Semi-supervised Small Apple Detection in Orchard Environments},
author={Johanson, Robert and Wilms, Christian and Johannsen, Ole and Frintrop, Simone},
booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
year = {2024}
}
  ```


