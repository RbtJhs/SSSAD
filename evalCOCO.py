import argparse
from shutil import copyfile
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def parse_args():
    parser = argparse.ArgumentParser('train net')
    parser.add_argument('--useSegm', dest='useSegm', type=str, default='False')
    parser.add_argument('--dataset', dest='dataset', type=str, default='data/MAD/annotations/instances_test.json')
    parser.add_argument('--results', dest='results', type=str, default='results/results_softteacher.json')

    args = parser.parse_args()
    args.useSegm = args.useSegm == 'True'
    return args

if __name__ == '__main__':
    args = parse_args()
    max_dets = [1, 100,100]

    cocoGt = COCO(args.dataset)

    cocoDt = cocoGt.loadRes(args.results)
    cocoEval = COCOeval(cocoGt, cocoDt)
    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
    cocoEval.params.maxDets = max_dets
    cocoEval.params.useSegm = args.useSegm
    cocoEval.params.useCats = False
    cocoEval.evaluate()
    cocoEval.accumulate()

    cocoEval.summarize()

