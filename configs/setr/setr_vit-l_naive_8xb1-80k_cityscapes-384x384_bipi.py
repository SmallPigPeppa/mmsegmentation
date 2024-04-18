_base_ = [
    '../_base_/models/setr_naive.py',
    '../_base_/datasets/cityscapes_384x384.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (384, 384)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        img_size=(384, 384),
        patch_size=8,
        drop_rate=0.,
        init_cfg=dict(
            type='Pretrained', checkpoint='pretrain/vit_large_p16.pth')),
    test_cfg=dict(mode='slide', crop_size=(384, 384), stride=(256, 256)))

optimizer = dict(weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
