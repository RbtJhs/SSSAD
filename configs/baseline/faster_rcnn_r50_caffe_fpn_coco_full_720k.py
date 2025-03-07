_base_ = "base.py"

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        ann_file="data/coco/annotations/instances_train2017.json",
        img_prefix="data/coco/train2017/",
    ),
)

optimizer = dict(lr=None)
lr_config = dict(step=[12000, 16000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=20000)
