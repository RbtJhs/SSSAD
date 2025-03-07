_base_="base.py"

data = dict(
    samples_per_gpu=20,
    workers_per_gpu=10,
    train=dict(

        sup=dict(

            ann_file="data/MAD_tiles/annotations/instances_train.json",
            img_prefix="data/MAD_tiles/train/",

        ),
        unsup=dict(

            ann_file="data/MAD_tiles/annotations/instances_unlabeled.json",
            img_prefix="data/MAD_tiles/unlabeled/",

        ),
    ),

    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[60000, 80000])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=100000)

