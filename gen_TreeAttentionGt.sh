echo 'Generate TreeAttention GT'

echo 'Train Split'
python gen_TreeAttentionGt.py --annotations "./data/MAD/annotations/instances_train.json" --save_to "./data/MAD/train_masks"

echo 'Val Split'
python gen_TreeAttentionGt.py --annotations "./data/MAD/annotations/instances_val.json" --save_to "./data/MAD/val_masks"

echo 'Test Split'
python gen_TreeAttentionGt.py --annotations "./data/MAD/annotations/instances_test.json" --save_to "./data/MAD/test_masks"
