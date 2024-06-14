_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=700, val_interval=50, dynamic_intervals=[(670, 1)])

auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3))

optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({'neck.encoder': dict(lr_mult=0.05)})),
    clip_grad=dict(max_norm=0.1, norm_type=2))

param_scheduler = [
    dict(
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=5,
        T_max=350,
        end=349,
        by_epoch=True,
        convert_to_iter_based=True),
    # this scheduler is used to increase the lr from 2e-4 to 5e-4
    dict(type='ConstantLR', by_epoch=True, factor=2.5, begin=349, end=350),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=350,
        T_max=320,
        end=670,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=670, end=700),
]

# data
input_size = (640, 640)
metafile = 'configs/_base_/datasets/crowdpose.py'
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)

train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type='YOLOXMixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='BottomupGetHeatmapMask', get_invalid=True),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

# data settings
data_mode = 'bottomup'
data_root = 'data/pose/CrowdPose/'

# mapping
aic_crowdpose = [(3, 0), (0, 1), (4, 2), (1, 3), (5, 4), (2, 5),
                 (9, 6), (6, 7), (10, 8), (7, 9), (11, 10), (8, 11), (12, 12),
                 (13, 13)]

coco_crowdpose = [
    (5, 0),
    (6, 1),
    (7, 2),
    (8, 3),
    (9, 4),
    (10, 5),
    (11, 6),
    (12, 7),
    (13, 8),
    (14, 9),
    (15, 10),
    (16, 11),
]

mpii_crowdpose = [
    (13, 0),
    (12, 1),
    (14, 2),
    (11, 3),
    (15, 4),
    (10, 5),
    (3, 6),
    (2, 7),
    (4, 8),
    (1, 9),
    (5, 10),
    (0, 11),
    (9, 12),
    (7, 13),
]

jhmdb_crowdpose = [(4, 0), (3, 1), (8, 2), (7, 3), (12, 4), (11, 5), (6, 6),
                   (5, 7), (10, 8), (9, 9), (14, 10), (13, 11), (2, 12),
                   (0, 13)]

halpe_crowdpose = [
    (5, 0),
    (6, 1),
    (7, 2),
    (8, 3),
    (9, 4),
    (10, 5),
    (11, 6),
    (12, 7),
    (13, 8),
    (14, 9),
    (15, 10),
    (16, 11),
]

posetrack_crowdpose = [
    (5, 0),
    (6, 1),
    (7, 2),
    (8, 3),
    (9, 4),
    (10, 5),
    (11, 6),
    (12, 7),
    (13, 8),
    (14, 9),
    (15, 10),
    (16, 11),
    (2, 12),
    (1, 13),
]

# train datasets
dataset_coco = dict(
    type='CocoDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='coco/train2017/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=14, mapping=coco_crowdpose)
    ],
)

dataset_aic = dict(
    type='AicDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(img='pose/ai_challenge/ai_challenger_keypoint'
                     '_train_20170902/keypoint_train_images_20170902/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=14, mapping=aic_crowdpose)
    ],
)

dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=14,
            mapping=[(i, i) for i in range(14)])
    ],
)

dataset_mpii = dict(
    type='MpiiDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='pose/MPI/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=14, mapping=mpii_crowdpose)
    ],
)

dataset_jhmdb = dict(
    type='JhmdbDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='jhmdb/annotations/jhmdb_train.json',
    data_prefix=dict(img='pose/JHMDB/images/'),
    pipeline=[
        dict(
            type='KeypointConverter', num_keypoints=14, mapping=jhmdb_crowdpose)
    ],
)

dataset_halpe = dict(
    type='HalpeDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='halpe/annotations/halpe_train.json',
    data_prefix=dict(img='pose/halpe/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=14,
            mapping=[(i, i) for i in range(14)])
    ],
)

dataset_posetrack = dict(
    type='PoseTrackDataset',
    data_root=data_root,
    data_mode=data_mode,
    ann_file='posetrack/annotations/posetrack_train.json',
    data_prefix=dict(img='pose/posetrack/posetrack_data/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=14,
            mapping=posetrack_crowdpose)
    ],
)

dataset = dict(
    type='MultiDataset',
    datasets=[
        dataset_coco,
        dataset_aic,
        dataset_crowdpose,
        dataset_mpii,
        dataset_jhmdb,
        dataset_halpe,
        dataset_posetrack,
    ])

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dataset,
    persistent_loader=True,
    persistent_sub_loaders=[
        dict(persistent_workers=True, num_workers=2),
        dict(persistent_workers=True, num_workers=2),
        dict(persistent_workers=True, num_workers=2),
        dict(persistent_workers=True, num_workers=2),
        dict(persistent_workers=True, num_workers=2),
        dict(persistent_workers=True, num_workers=2),
        dict(persistent_workers=True, num_workers=2),
    ],
)

# val datasets
val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dataset_crowdpose)

# test datasets
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(type='CocoMetric', ann_file=data_root +
                     'annotations/mmpose_crowdpose_trainval.json')

test_evaluator = val_evaluator

# visualizer settings
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend')
    ],
    name='visualizer'
)

# run settings
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    timer=dict(type='IterTimerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook'))

log_level = 'INFO'
load_from = None
resume = None
work_dir = 'work_dirs/pose_estimation_finetune'
workflow = [('train', 1), ('val', 1)]
launcher = 'none'
widen_factor = 1.0
