'''
contrastive learning finetuning ucf
'''
from yxy import path
################## ! essentials ###################
expname = 'finetuning'
arch = 'c3d'  # 'resnet34''resnet18''s3dg'
work_dir = f'{path.work}/cepworkspace/{expname}'
total_epochs = 50
gpu_batch = 16
load_model = None  # the pretrained model path
load_checkpoint = None
validate = False
only_train_fc = False  # is linearprob

################## ! model ###################
model = dict()

################## !dataset ###################

finetune_data = dict(
    dataset=dict(
        name='ucf101',
        root=f'{path.data}/ucf101/UCF-101',
        annotation_path=f'{path.data}/ucf101/split',
        fold=1,  # validate training split fold
        mean=[0.485, 0.456, 0.406],  # imagenet Norm mean
        std=[0.229, 0.224, 0.225],  # imagenet Norm mea
        num_classes=101,
    ),
    # fintuning on HMDB
    # dataset=dict(
    #     name='hmdb51',
    #     root=f'{path.data}/hmdb51/video',
    #     annotation_path=f'{path.data}/hmdb51/split',
    #     fold=1,  # validate training split fold
    #     mean=[0.485, 0.456, 0.406],  # imagenet Norm mean
    #     std=[0.229, 0.224, 0.225],  # imagenet Norm mea
    #     num_classes=51,
    # ),
    spatial_transforms=dict(
        size=112,
        crop_area=dict(min=0.25, max=1.0),
        color_jitter=dict(brightness=0, contrast=0, saturation=0, hue=0),
        h_flip=0.5,
        gray_scale=0,
    ),
    temporal_transforms=dict(
        type='clip',
        size=16,  # frame needed
        strides=[{
            'stride': 1,
            'weight': 1
        }],
        validate=dict(stride=1, final_n_crop=10, n_crop=1),
        frame_rate=25,
    ),
    batch_size=gpu_batch,
    validate=dict(batch_size=16),
    final_validate=dict(batch_size=2),
    num_workers=2,
)

################## !optimizer ###################
# SGD
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    dampening=0,
    weight_decay=1e-4,
    nesterov=False,
)
# adam
# optimizer = dict(
#     type='Adam',
#     lr=1e-2,
#     eps=1e-4
# )
#

################## ! scheduler ###################

# scheduler = dict(
#     type='plateau',
#     patience=None,
# )
#
# scheduler = dict(
#     type='multistep',
#     milestones=None,
# )

scheduler = dict(type='cosine', )

# scheduler = dict(type='none', )

################## ! others ###################
log_level = 'INFO'
dist_params = dict(backend='nccl')
log_interval = 1
no_scale_lr = False

checkpoint_config = dict(keep_interval=1)  # save for each epoch
