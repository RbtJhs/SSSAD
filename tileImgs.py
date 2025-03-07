import glob
import numpy as np
from skimage.io import imread,imsave
import json
import os
import io
from numpyencoder import NumpyEncoder
import shutil
import math
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Pool
import tqdm
import torch
import torch.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Cut images into overlapping tiles")

    parser.add_argument("--workers", type=int, default=8, help="number fo workers for mp")

    parser.add_argument("--load_imgs_from",type=str, default="./data/MAD/val", help="path to load folder")
    parser.add_argument("--load_anns_from",type=str, default="./data/MAD/annotations/instances_val.json", help="path to load anns from")
    parser.add_argument("--save_imgs_to",type=str, default="./data/MAD_tiles/val", help="path to save folder")
    parser.add_argument("--save_anns_to",type=str, default="./data/MAD_tiles/annotations", help="path to save anns")
    parser.add_argument("--save_anns_name",type=str, default="./data/MAD_tiles/annotations/instances_val.json", help="path to save anns")

    parser.add_argument("--window_size", type=list, default=[800,800], help="size of crop window")
    parser.add_argument("--stride", type=float, default=0.5, help="stride of crop window")
    args = parser.parse_args()
    return args


def computePadding(img_shape,kernel_shape,stride):
    h,w,c = img_shape
    wh,ww = kernel_shape
    tiles_h = (h % (wh*stride))
    tiles_w = (w % (ww * stride))
    ph = (wh*stride)- tiles_h
    pw = (ww * stride) - tiles_w
    return int(math.ceil(ph/2)),int(math.ceil(pw/2))



def boxes2Tensor(canvas_shape,boxes):
    #make a tensor with w x h x num_boxes, for easy slicing (note: very memory inefficient)
    box_tensor = np.zeros((canvas_shape[0],canvas_shape[1],len(boxes)),dtype=bool)
    areas = []
    for i,box in enumerate(boxes):
        x,y,h,w = box
        areas.append(w*h)
        box_tensor[y:y+w, x:x+h, i] = 1

    return torch.tensor(box_tensor).permute(2,0,1),areas

def Tensor2Boxes(box_tensor,areas):
    #retransform tensor back to boxes
    boxes = []
    for i in range(box_tensor.shape[2]):
        box_tile = box_tensor[:,:,i]
        OG_area= float(areas[i])
        if(np.max(box_tile)!=0):
            a = np.where(box_tile != 0)
            bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
            new_area = float((bbox[1] - bbox[0])*(bbox[3] - bbox[2]))
            area_ratio = new_area/OG_area
            if(area_ratio > 0.2):
                boxes.append(bbox)
    return boxes

def makeCOCOAnn(imageId, boxes):
    #create coco annotations for each bbox in tile
    annotations = []
    for box in boxes:
        annDict = {}

        h = box[1]-box[0]
        w = box[3]-box[2]
        area = h*w
        annDict['category_id'] = 1
        annDict['area'] = area
        annDict['id'] = 0
        annDict['image_id'] = imageId
        annDict['iscrowd'] = False
        annDict['bbox'] = [box[2], box[0], w, h]
        annotations.append(annDict)
    return annotations


def loadAnns(args):
    '''
    assign each annotation to its corresponding image and add the filenames (tile ranges) for each
    tile to the dict, since we use unordered mp this is usefull for later image reconstruction from tiles

    :param args:
    :return: dict containing information for tiling
    '''
    with open(args.load_anns_from) as data_file:  # annotation file
        anns = json.loads(data_file.read())
    ih,iw = anns["images"][0]["height"], anns["images"][0]["width"]
    ph, pw = computePadding([ih,iw,3], args.window_size, args.stride)
    ann_dict = {}
    num_tiles = int((((ih+ (2*ph))/(args.window_size[0]*args.stride))-1) * (((iw+ (2*pw))/(args.window_size[1]*args.stride))-1))
    tile_count = 0
    for img_info in anns["images"]:

        ann_dict[img_info["id"]] = {}
        ann_dict[img_info["id"]]["anns"] = []
        ann_dict[img_info["id"]]["tile_range"] = list(range(tile_count,tile_count+num_tiles))
        ann_dict[img_info["id"]]["args"] = args
        tile_count += num_tiles
    for ann in anns["annotations"]:
        ann_dict[ann["image_id"]]["anns"].append(ann["bbox"])
    return ann_dict




def tile_img(data):
    imgID,img_data = data
    boxes,tile_range,args = img_data["anns"],img_data["tile_range"],img_data["args"]
    img_path =args.load_imgs_from +"/" + str(imgID).zfill(12) + ".jpg"
    img = imread(img_path)

    imgInfos = []
    anns = []
    tw,th = args.window_size

    ph, pw = computePadding(img.shape, (tw, th), args.stride)
    boxes = np.asarray(boxes)
    boxes[:, 0] = boxes[:, 0] + pw
    boxes[:, 1] = boxes[:, 1] + ph

    img = np.pad(img, ((ph, ph), (pw, pw), (0, 0)), mode='constant', constant_values=0)
    box_tensor,areas = boxes2Tensor(img.shape,boxes)

    tw,th = args.window_size
    stride_w,stride_h = int(tw*args.stride),int(th*args.stride)
    h,w,c = img.shape
    ys,xs = torch.arange(0,h-stride_h,step=stride_h) , torch.arange(0,w-stride_w,step=stride_w)
    X,Y = xs + ys[:,None]*0 , xs*0 + ys[:,None]
    #coords = torch.dstack((X,Y)).flatten(0,1).numpy()
    coords = torch.dstack((X-pw,Y-ph)).flatten(0,1).numpy()

    img_patches = torch.tensor(img).unfold(0, tw,stride_w).unfold(1, th,stride_h)
    img_patches = img_patches.flatten(0,1).permute(0,2,3,1).numpy()

    box_patches = box_tensor.unfold(1, tw,stride_w).unfold(2, tw,stride_h)
    box_patches = box_patches.flatten(1,2).permute(1,2,3,0).numpy()

    for i,(img_patch,box_patch,pos) in enumerate(zip(img_patches,box_patches,coords)):
        boxes4crop = Tensor2Boxes(box_patch,areas)
        tileID = tile_range[i]
        tile_name = str(tileID).zfill(12) + ".jpg"
        imgInfo = {}
        imgInfo['height'] = th
        imgInfo['width'] = tw
        imgInfo['file_name'] = tile_name
        imgInfo['file_name_untiled'] = str(imgID).zfill(12) + ".jpg"
        imgInfo['id'] = tileID
        imgInfo['id_untiled'] = imgID
        imgInfo['pos_offset'] = pos
        imgInfo['padding'] = [ph,pw]
        imgInfo['original_size'] = [h-(2*ph),w-(2*pw)]
        imgInfos.append(imgInfo)

        if (len(boxes4crop) != 0):
            ann = makeCOCOAnn(tileID, boxes4crop)
            anns += ann
        imsave(args.save_imgs_to + '/' + tile_name, img_patch)

    return anns,imgInfos

if __name__ == '__main__':
    args = parse_args()

    if(not os.path.exists("./data/MAD_tiles")):
        os.mkdir("./data/MAD_tiles")
    if(not os.path.exists(args.save_imgs_to)):
        os.mkdir(args.save_imgs_to)
    if(not os.path.exists(args.save_anns_to)):
        os.mkdir(args.save_anns_to)



    anns_untiled = loadAnns(args)
    new_anns = []
    img_infos = []
    pool = Pool(args.workers)                         # Create a multiprocessing Pool
    for n_ann,n_info in tqdm.tqdm(pool.imap_unordered(tile_img, anns_untiled.items()), total=len(anns_untiled.keys())):
        new_anns += n_ann
        img_infos += n_info

    for i in range(len(new_anns)):
        new_anns[i]["id"] = i

    dataDict = {}
    dataDict['annotations'] = new_anns
    dataDict['images'] = img_infos
    dataDict['categories'] = [{u'supercategory': u'tree', u'id': 1, u'name': u'apple' }]

    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    with io.open(args.save_anns_name, 'w', encoding='utf-8') as f:
        str_ = json.dumps(dataDict, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False,cls=NumpyEncoder)
        f.write(to_unicode(str_))
