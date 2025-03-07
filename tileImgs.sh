echo 'Tile MAD Dataset'

echo 'Train Split'
python tileImgs.py --load_imgs_from "./data/MAD/train" --load_anns_from "./data/MAD/annotations/instances_train.json" --save_imgs_to "./data/MAD_tiles/train" --save_anns_to "./data/MAD_tiles/annotations" --save_anns_name "./data/MAD_tiles/annotations/instances_train.json"

echo 'Val Split'
python tileImgs.py --load_imgs_from "./data/MAD/val" --load_anns_from "./data/MAD/annotations/instances_val.json" --save_imgs_to "./data/MAD_tiles/val" --save_anns_to "./data/MAD_tiles/annotations" --save_anns_name "./data/MAD_tiles/annotations/instances_val.json"

echo 'Test Split'
python tileImgs.py --load_imgs_from "./data/MAD/test" --load_anns_from "./data/MAD/annotations/instances_test.json" --save_imgs_to "./data/MAD_tiles/test" --save_anns_to "./data/MAD_tiles/annotations" --save_anns_name "./data/MAD_tiles/annotations/instances_test.json"

echo 'Unlabeled Split'
python tileImgsUnlabeled.py --load_imgs_from "./data/MAD/unlabeled" --save_imgs_to "./data/MAD_tiles/unlabeled" --save_anns_to "./data/MAD_tiles/annotations" --save_anns_name "./data/MAD_tiles/annotations/instances_unlabeled.json"
