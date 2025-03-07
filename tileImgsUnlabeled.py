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
import glob
import cv2
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Cut images into overlapping tiles")

    parser.add_argument("--workers", type=int, default=12, help="number fo workers for mp")

    parser.add_argument("--load_imgs_from",type=str, default="./data/MAD/unlabeled", help="path to load folder")
    parser.add_argument("--save_imgs_to",type=str, default="./data/MAD_tiles/unlabeled", help="path to save folder")
    parser.add_argument("--save_anns_to",type=str, default="./data/MAD_tiles/annotations", help="path to save anns")
    parser.add_argument("--save_anns_name",type=str, default="./data/MAD_tiles/annotations/instances_unlabeled.json", help="path to save anns")
    parser.add_argument("--img_size", type=list, default=[3840,2160], help="not all unlabeled images have the same size")
    parser.add_argument("--window_size", type=list, default=[800,800], help="size of sliding window")
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


def loadData(args):
    '''
    assign each annotation to its corresponding image and add the filenames (tile ranges) for each
    tile to the dict, since we use unordered mp this is usefull for later image reconstruction from tiles

    :param args:
    :return: dict containing information for tiling
    '''
    paths = glob.glob(args.load_imgs_from + "/*")
    iw,ih = args.img_size
    ph, pw = computePadding([ih,iw,3], args.window_size, args.stride)
    ann_dict = {}
    num_tiles = int((((ih+ (2*ph))/(args.window_size[0]*args.stride))-1) * (((iw+ (2*pw))/(args.window_size[1]*args.stride))-1))
    tile_count = 0
    for path in paths:
        name = int(path.split("/")[-1].split(".")[0])
        ann_dict[name] = {}
        ann_dict[name]["tile_range"] = list(range(tile_count,tile_count+num_tiles))
        ann_dict[name]["args"] = args
        tile_count += num_tiles
        cw,ch =  Image.open(path).size
        ann_dict[name]["resize"] = not((ih==ch) and (iw==cw))


    return ann_dict


def resize_img(args,img):
    ih,iw,ic = img.shape
    if (ih > iw):
        return cv2.resize(img,(args.img_size[1],args.img_size[0]))
    else:
        return cv2.resize(img,args.img_size)



def tile_img(data):
    imgID,img_data = data
    tile_range,args = img_data["tile_range"],img_data["args"]
    img = imread(args.load_imgs_from +"/" + str(imgID).zfill(12) + ".jpg")
    if(img_data["resize"]):
        img = resize_img(args,img)

    imgInfos = []

    tw,th = args.window_size
    ph, pw = computePadding(img.shape, (tw, th), args.stride)
    img = np.pad(img, ((ph, ph), (pw, pw), (0, 0)), mode='constant', constant_values=0)

    stride_w,stride_h = int(tw*args.stride),int(th*args.stride)
    h,w,c = img.shape
    ys,xs = torch.arange(0,h-stride_h,step=stride_h) , torch.arange(0,w-stride_w,step=stride_w)
    X,Y = xs + ys[:,None]*0 , xs*0 + ys[:,None]
    coords = torch.dstack((X,Y)).flatten(0,1).numpy()

    img_patches = torch.tensor(img).unfold(0, tw,stride_w).unfold(1, th,stride_h)
    img_patches = img_patches.flatten(0,1).permute(0,2,3,1).numpy()


    for i,(img_patch,pos) in enumerate(zip(img_patches,coords)):
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
        imgInfos.append(imgInfo)
        
        imsave(args.save_imgs_to + '/' + tile_name, img_patch)

    return imgInfos

if __name__ == '__main__':
    args = parse_args()


    if(not os.path.exists(args.save_imgs_to)):
        os.mkdir(args.save_imgs_to)
    if(not os.path.exists(args.save_anns_to)):
        os.mkdir(args.save_anns_to)


    anns_untiled = loadData(args)
    img_infos = []
    pool = Pool(args.workers)                         # Create a multiprocessing Pool
    for n_info in tqdm.tqdm(pool.imap_unordered(tile_img, anns_untiled.items()), total=len(anns_untiled.keys())):
        img_infos += n_info


    dataDict = {}
    dataDict['annotations'] = []
    dataDict['images'] = img_infos
    dataDict['categories'] = [{u'supercategory': u'tree', u'id': 1, u'name': u'apple' }]

    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    with io.open(args.save_anns_name, 'w', encoding='utf-8') as f:
        str_ = json.dumps(dataDict, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False,cls=NumpyEncoder)
        f.write(to_unicode(str_))

