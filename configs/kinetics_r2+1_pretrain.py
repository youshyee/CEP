'''
#   ____ _____ ____
#  / ___| ____|  _ \
# | |   |  _| | |_) |
# | |___| |___|  __/
#  \____|_____|_|
#
'''
from yxy import path
import mmcv
################## ! essentials ###################
expname = 'k400_r2+1'
work_dir = f'{path.work}/cepworkspace/{expname}'
total_epochs = 50
gpu_batch = 2
load_model = None
load_checkpoint = None

validate = False

################## ! model ###################
arch = 'r2plus1d'
model = dict()

################## ! contrastive  ###################
contrastive = dict(
    dim=2048,
    t=1,
    k=16384,
    m=0.9,
    fc_type='mlp',
    projectorbn=[False, True],
)

################## !dataset ###################
pretrain_data = dict(
    dataset=dict(
        name='kinetics400',
        root=f'{path.data}/kinetics400',
        blacklist=mmcv.list_from_file(f'{path.data}/kinetics400/blacklist'),
        mean=[0.485, 0.456, 0.406],  # imagenet Norm mean
        std=[0.229, 0.224, 0.225],  # imagenet Norm mea
    ),
    aug_plus=False,
    spatial_transforms=dict(size=224),
    temporal_transforms=dict(
        size=16*3,
        strides=[{
            'stride': 1,
            'weight': 1
        }],
        frame_rate=24,
    ),  # frame needed
    batch_size=gpu_batch,
    num_workers=4,
)


# loss
loss_lambda = dict(A=1.0, CL=1., CC=0.1)

################## !optimizer ###################
optimizer = dict(
    type='SGD',
    lr=1e-2,
    momentum=0.9,
    dampening=0,
    weight_decay=1e-4,
    nesterov=False,
)

################## ! others ###################
log_level = 'INFO'
dist_params = dict(backend='nccl')
log_interval = 1
no_scale_lr = False

checkpoint_config = dict(keep_interval=1)  # save for each epoch
