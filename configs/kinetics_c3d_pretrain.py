'''
contrastive learning basic target baseline
'''
from yxy import path  # personal path manager
import mmcv
################## ! essentials ###################
expname = 'k400_c3d'
work_dir = f'{path.work}/cepworkspace/{expname}'
total_epochs = 200
gpu_batch = 8
load_model = None
load_checkpoint = None
validate = False

################## ! model ###################
arch = 'c3d'
model = dict()

################## ! contrastive  ###################
contrastive = dict(
    dim=128,
    t=0.07,
    k=16384,
    m=0.999,
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
        size=16,
        strides=[{
            'stride': 1,
            'weight': 1
        }],
        frame_rate=20,
    ),  # frame needed
    batch_size=gpu_batch,
    num_workers=4,
)

# loss
loss_lambda = dict(A=1.0, CL=10., CC=10., CC_type='l2')

################## !optimizer ###################
optimizer = dict(
    type='SGD',
    lr=1e-1,
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
