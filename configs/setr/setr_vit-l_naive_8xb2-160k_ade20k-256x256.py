_base_ = [
    '../_base_/models/setr_naive.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        img_size=(256, 256),
        patch_size=16,
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/vit_large_p16.pth')),
    decode_head=dict(num_classes=150),
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=150,
            dropout_ratio=0,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)),
)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='Resize', scale=(1024, 256), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')]

dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

optimizer = dict(lr=0.01, weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
