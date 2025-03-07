import glob
import numpy as np
from skimage.io import imread
import json
from skimage.transform import resize
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--annotations", default="./data/MAD/annotations/instances_test.json", help="Path to annotation file")
    parser.add_argument("--gt", default="./data/MAD/test_masks", help="Path to TreeAttention gt")

    parser.add_argument("--predictions", default="./results/TreeAttention/test", help="Path to TreeAttention predictions")


    args = parser.parse_args()
    return args


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


def computeIoU(pred,mask):
    intersection = np.sum(np.logical_and(pred,mask))
    union = np.sum(np.logical_or(pred,mask))
    return intersection/union



if __name__ == '__main__':
    args = parse_args()
    data_dict = loadJSON(args.annotations)

    meanIou = []
    mean_gt_included = []

    for key in tqdm(data_dict.keys(),total=len(data_dict.keys())):
        boxes = data_dict[key]["anns"]
        h,w = data_dict[key]["shape"]
        pred = imread(args.predictions + "/" + str(key).zfill(12) + ".png")
        pred = resize(pred, (h, w))
        gt = imread(args.gt + "/" + str(key).zfill(12) + ".jpg")
        gt = resize(gt, (h, w))

        pred[pred >= 0.3]=1
        pred[pred < 0.3] = 0

        counter = 0
        for box in boxes:
            x,y,bw,bh = box
            area = bw*bh
            pred_area = pred[y:y+bh,x:x+bw]
            overlap = np.sum(pred[y:y+bh,x:x+bw]) / area
            if(overlap > 0.5):
                counter += 1
        mean_gt_included.append(counter/len(boxes))
        iou = computeIoU(pred,gt)
        meanIou.append(iou)
    print("mIoU: ", np.mean(meanIou))
    print("avg number boxes included in bin mask:",np.mean(mean_gt_included))

