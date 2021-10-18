# CEP
Official implementation for paper "Back to the Future: Cycle Encoding Prediction for Self-supervised Video Representation Learning"


![overview](https://youshyee.github.io/CEP/overview.jpg)


## Getting Started
### Install required packages

All dependencies can be installed using pip:

```sh
pip install -r requirements.txt
```

### Datasets

Data for pre-training ([Kinetics-400](https://deepmind.com/research/open-source/kinetics)) and fine-tuning ([UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)).

After the Download, organise the dataset directory hierarchy is as follow:
```
├── data
    ├── kinetics400
    │   ├── train_video
    │   │   ├── answering_questions
    │   │   │   └── *.mp4
    │   │   └── ...
    │   └── val_video
    │       └── (same as train_video)
    ├── UCF101
    │   ├── ucfTrainTestlist
    │   │   ├── classInd.txt
    │   │   ├── testlist01.txt
    │   │   ├── trainlist01.txt
    │   │   └── ...
    │   └── UCF-101
    │       ├── ApplyEyeMakeup
    │       │   └── *.avi
    │       └── ...
    ├── hmdb51
    │   ├── metafile
    │   │   ├── brush_hair_test_split1.txt
    │   │   └── ...
    │   └── videos
    │       ├── brush_hair
    │       │   └── *.avi
    │       └── ...

```

## Self-supervised pre-training

Modify the data path, workdir path in the config files and train model with,

```sh

# Arch: ResNet-18
sh dist_train.sh $GPU_NUM  configs/kinetics_r18_pretrain.py

# Arch: ResNet-34
sh dist_train.sh $GPU_NUM  configs/kinetics_r34_pretrain.py

# Arch: S3D-G
sh dist_train.sh $GPU_NUM  configs/kinetics_s3dg_pretrain.py

# Arch: Slowfast
sh dist_train.sh $GPU_NUM  configs/kinetics_slowfast_pretrain.py

# Arch: R(2+1)D
sh dist_train.sh $GPU_NUM  configs/kinetics_r2+1_pretrain.py
```

$GPU_NUM is the number of gpus used for distributed training

### Training in Cluster
```sh

sbatch ./cep_slurm.sh

```

