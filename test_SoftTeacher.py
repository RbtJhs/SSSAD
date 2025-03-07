import json
import numpy as np
from skimage.io import imread, imsave
import cv2
import io
from torchvision.ops import nms
import torch
from numpyencoder import NumpyEncoder
from skimage.transform import resize
import math
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from mmdet.apis import inference_detector
from ssod.apis.inference import init_detector
from ssod.utils import patch_config
from mmcv import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Cut images into overlapping tiles")

    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument("--config",default="./configs/soft_teacher/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k.py", help="Config file")
    parser.add_argument("--checkpoint",default="./work_dirs/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_full_720k_current_best/iter_115000.pth", help="Checkpoint file")
    parser.add_argument("--batch", type=int, default=30, help="Batch size")

    parser.add_argument("--load_imgs_from",type=str, default="./data/MAD/test", help="path to load folder")
    parser.add_argument("--load_tiles_from",type=str, default="./data/MAD_tiles/test", help="path to load folder")

    parser.add_argument("--load_masks_from",type=str, default="./results/TreeAttention/test", help="path to load folder")

    parser.add_argument("--load_anns_from",type=str, default="./data/MAD_tiles/annotations/instances_test.json", help="path to load anns from")

    parser.add_argument("--window_size", type=list, default=[800,800], help="size of crop window")
    parser.add_argument("--stride", type=float, default=0.5, help="stride of crop window")


    parser.add_argument("--save_results_to",type=str, default="./results/test_results_best2.json", help="path to save anns")
    parser.add_argument("--nms_th",type=float, default=0.5, help="threshold for NMS")
    parser.add_argument("--score_th",type=float, default=0.0, help="threshold for confidence score")

    parser.add_argument("--vis",type=bool, default=False, help="vis results")
    parser.add_argument("--save_vis_to",type=str, default="./results/vis/", help="path to save anns")


    args = parser.parse_args()
    return args


def load_detector_model(args):
    #print(args)
    cfg = Config.fromfile(args.config)
    #print(cfg.test_pipeline.MultiScaleFlipAug)
    # Not affect anything, just avoid index error

    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    return model



def loadJSON(path):
    with open(path) as data_file:  # annotation file
        anns = json.loads(data_file.read())
    return anns




def genBorderDict_dynamic(border,thickness,tilesize):
    border_dict = {}
    stride = int(tilesize/2)
    b1 =border.copy()
    b1[0:stride, 0:thickness] = 1
    b1[0:thickness, 0:stride] = 1
    border_dict['-1_-1'] = b1.nonzero()

    b2 =border.copy()
    b2[stride:tilesize, 0:thickness] = 1
    b2[tilesize-thickness:tilesize, 0:stride] = 1
    border_dict['1_-1'] = b2.nonzero()

    b3 =border.copy()
    b3[0:stride, tilesize-thickness:tilesize] = 1
    b3[0:thickness, stride:tilesize] = 1
    border_dict['-1_1'] = b3.nonzero()

    b4 =border.copy()
    b4[stride:tilesize, tilesize-thickness:tilesize] = 1
    b4[tilesize-thickness:tilesize, stride:tilesize] = 1
    border_dict['1_1'] = b4.nonzero()

    b5 =border.copy()
    b5[0:thickness, :] = 1
    border_dict['-1_0'] = b5.nonzero()

    b6 =border.copy()
    b6[tilesize-thickness:tilesize, :] = 1
    border_dict['1_0'] = b6.nonzero()

    b7 =border.copy()
    b7[:, 0:thickness] = 1
    border_dict['0_-1'] = b7.nonzero()

    b8 =border.copy()
    b8[:, tilesize-thickness:tilesize] = 1
    border_dict['0_1'] = b8.nonzero()


    return border_dict



def compute_border(pos,border_dict,border,matrix):
    xidx, yidx = pos
    mh,mw = matrix.shape
    b = border.copy()
    for i in range(-1,2):
        for j in range(-1,2):
            if(yidx+i < 0 or xidx+j < 0 or yidx+i >= mh or xidx+j >= mw):
                continue
            if((i!=0) or (j!= 0)):
                if(matrix[yidx+i,xidx+j]):
                    b[border_dict[str(i)+'_'+str(j)]] = 1

    return b.astype(int)

def computePadding(img_shape,kernel_shape,stride):
    h,w,c = img_shape
    wh,ww = kernel_shape
    ph = (wh*stride)-(h % (wh*stride))
    pw = (ww * stride) - (w % (ww * stride))
    return int(math.ceil(ph/2)),int(math.ceil(pw/2))

def save_img(args,imgID,boxes):
    img = imread(args.load_imgs_from +"/" + str(imgID).zfill(12)+".jpg")
    if(not os.path.exists(args.save_vis_to)):
        os.mkdir(args.save_vis_to)
    boxes = boxes[boxes[:, 4] >= 0.5]
    for box in boxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (245, 203, 17), 2)

    imsave(args.save_vis_to + "/"+ str(imgID).zfill(12)+".jpg",img)


def getTilePaths(args,tile_Ids):
    paths = []

    for id in tile_Ids:
            paths.append(args.load_tiles_from + "/" + str(id).zfill(12) + ".jpg")

    batches = [paths[n:n+args.batch] for n in range(0, len(paths), args.batch)]
    return batches

def reconstruct(args,model,imgID,values,border_dict,border):
    tw,th = args.window_size
    stride_w,stride_h = int(tw*args.stride),int(th*args.stride)
    value_keys = sorted(values, key=lambda k: k)   #sort tileIds just in case

    img_info = values[value_keys[0]]["img_info"]
    mask = imread(args.load_masks_from +"/"+str(imgID).zfill(12)+".png")

    ph, pw = img_info["padding"]

    mask = resize(mask,(img_info["original_size"]),anti_aliasing=False)
    mask = np.pad(mask, ((ph, ph), (pw, pw)), mode='symmetric')
    mask_patches = torch.tensor(mask).unfold(0, tw,stride_w).unfold(1, th,stride_h).numpy()
    patch_mean = np.mean(mask_patches,axis=(2,3))
    patches_keep = patch_mean > 0.2

    tile_Ids_keep = np.asarray(value_keys)[patches_keep.flatten()]

    batches = getTilePaths(args, tile_Ids_keep)
    results = []
    for batch in batches:
        result = inference_detector(model,batch)
        results += result

    all_boxes = []
    for i,id in enumerate(tile_Ids_keep):
        px,py = values[id]["img_info"]["pos_offset"]
        boxes = np.asarray(results[i][0])
        if(len(boxes) == 0):

            continue

        border_tile = compute_border([int((px+pw)/stride_w),int((py+ph)/stride_h)],border_dict,border,patches_keep)

        #remove boxes with weird ratios
        boxes_w = boxes[:,2] - boxes[:,0]
        boxes_h = boxes[:,3] - boxes[:,1]
        ratio = (((boxes_w/(boxes_h+0.0000001)) <= 5) & ((boxes_h/(boxes_w+0.0000001)) <= 5))
        boxes = boxes[ratio]

        #remove boxes that are on the tile border
        areas = boxes_w[ratio]*boxes_h[ratio]
        areas_border = []
        for xs,ys,xe,ye,score in boxes.astype(int):
            areas_border.append(np.sum(border_tile[ys:ye,xs:xe]))

        overlap = np.asarray(areas_border)/areas
        boxes = boxes[overlap < 0.95]

        #remove boxes that are in padded area
        boxes = boxes + np.asarray([px,py,px,py,0])
        boxes = boxes[boxes[:,1] >= 0]
        boxes = boxes[boxes[:,3] >= 0]

        boxes = boxes[boxes[:,4] >= args.score_th] #remove boxes under confidence th

        all_boxes += list(boxes)

    all_boxes = np.asarray(all_boxes)
    all_boxes = torch.tensor(np.asarray(all_boxes)) #.to(device)
    keep= nms(all_boxes[:,:4],all_boxes[:,4],args.nms_th)
    boxes_keep = all_boxes[keep].detach().cpu().numpy()

    if(args.vis):
        save_img(args,imgID,boxes_keep)

    coco_results = []
    for box in boxes_keep:
        s = float(box[4])
        w = int(box[2]-box[0])
        h = int(box[3]-box[1])
        results_coco_format = {}
        results_coco_format['image_id']= int(imgID)
        results_coco_format['bbox'] = [int(box[0]),int(box[1]),w,h]
        results_coco_format['score'] = s
        results_coco_format['category_id'] = 1
        coco_results.append(results_coco_format)


    return coco_results



def prepare_data(args):

    anns = loadJSON(args.load_anns_from)


    img_infos = anns['images']
    Img_tile_map = {}

    for img_info in img_infos:
        id = img_info["id_untiled"]
        if(id not in Img_tile_map):
            Img_tile_map[id] = {}

    for img_info in img_infos:
        Img_tile_map[img_info["id_untiled"]][img_info["id"]] = {}
        Img_tile_map[img_info["id_untiled"]][img_info["id"]]["img_info"] = img_info


    return Img_tile_map


def main(args):

    model = load_detector_model(args)
    border = np.zeros(args.window_size,dtype=np.uint8)
    border_dict = genBorderDict_dynamic(border,int(args.window_size[0]*0.2),args.window_size[0])
    Img_tile_map = prepare_data(args)
    outputs_cocoeval = []
    for (imgID,values) in tqdm(Img_tile_map.items(),total=len(Img_tile_map.keys())):
        outputs_cocoeval += reconstruct(args,model,imgID,values,border_dict,border)



    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    with io.open(args.save_results_to, 'w', encoding='utf-8') as f:
        str_ = json.dumps(outputs_cocoeval, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False,cls=NumpyEncoder)
        f.write(to_unicode(str_))

if __name__ == '__main__':
    args = parse_args()

    device = torch.device(args.device)
    main(args)
