from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import Delaunay
import numpy as np
import os
import argparse
import json
from skimage.transform import resize
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--annotations", default="./MAD/annotations/instances_test.json", help="Path to annotation file")
    parser.add_argument("--save_to",default="./MAD/test_masks", help="path to save dir")
    parser.add_argument("--alpha", type=int, default=50, help="Alpha for alphashape")


    args = parser.parse_args()
    return args



def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def fill_contours(arr):
    return np.maximum.accumulate(arr, 1) &\
           np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1] &\
           np.maximum.accumulate(arr[::-1, :], 0)[::-1, :] &\
           np.maximum.accumulate(arr, 0)




def attMap2Mask(bin_mask,args):

    x, y = np.where(bin_mask != 0)
    points = np.column_stack((y, x))
    # Computing the alpha shape
    edges = list(alpha_shape(points, alpha=args.alpha, only_outer=True))
    mask = bin_mask * 0
    for i, j in edges:
        xx = points[[i, j], 0]
        yy = points[[i, j], 1]
        xy1 = [xx[0], yy[0]]
        xy2 = [xx[1], yy[1]]
        mask = cv2.line(mask, xy1, xy2, (255), thickness=1, lineType=8)

    return fill_contours(mask.copy().astype(int)),mask


def loadJSON(path):
    with open(path) as data_file:  # annotation file
        anns = json.loads(data_file.read())
    ann_dict = {}
    for img_info in anns["images"]:
        ann_dict[img_info["id"]] = {}
        ann_dict[img_info["id"]]["anns"] = []

        ann_dict[img_info["id"]]["shape"] = [img_info["height"],img_info["width"]]

    for ann in anns["annotations"]:
        ann_dict[ann["image_id"]]["anns"].append(ann["bbox"])

    return ann_dict



if __name__ == '__main__':
    args = parse_args()
    anns = loadJSON(args.annotations)
    if(not os.path.exists(args.save_to)):
        os.mkdir(args.save_to)
    for key in tqdm(anns.keys(),total=len(anns.keys())):
        h,w = anns[key]["shape"]
        boxes = anns[key]["anns"]
        mask = np.zeros([h,w])
        for box in boxes:
            x,y,bw,bh = box
            mask[y:y+bh,x:x+bw] = 1
        if (h > w):
            hn = 512
            wn = int((512 / h) * w)
        else:
            wn = 512
            hn = int((512 / w) * h)
        mask  = resize(mask, (hn, wn))

        mask,con = attMap2Mask(mask,args)
        cv2.imwrite(args.save_to + '/'+str(key).zfill(12)+'.jpg', mask.astype(np.uint8))

        #imsave(args.save_to + '/'+str(key).zfill(12)+'.jpg',mask.astype(np.uint8))


