# 2D CT Image Based Non-small Cell Lung Cancer (NSCLC) Classification Using Convolutional Neural Networks

**This repository contains the source code of paper *2D CT Image Based Non-small Cell Lung Cancer (NSCLC) Classification Using Convolutional Neural Networks*.**

## About this repository
- *train folder:* 
  - *models folder:*
    - Baseline model
    - Image-wise model
    - Patch-wise model
    - ResNet model
    - **CatNet model (proposed)**
  - the dataset(csv format): *Lung1.clinical.csv*
  - the code for training and testing the model: *generate_val_set.py, sequence_folders.py, train.py, utils.py*
  - list of samples for testing: *val_list.txt*
  
- preprocess the dataset: *prepare_training_data.py*

**Notice that the entire dataset (2D CT images) can be downloaded [here](https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics#fa40518ae0634edca8cfa8e1df141bda).**

## Prerequisites
```
torchvision
scipy
argparse
tensorboardX
blessings
progressbar2
path.py
matplotlib
opencv-python
scikit-image
pypng
tqdm
spatial-correlation-sampler
pandas
pytorch
numpy
```
Train in the Ubuntu environment, on a GeForce RTX 2080Ti GPU. 

## Prepare data for training
The CT files can be converted into jpg images by running the command:
```
python prepare_training_data.py  \
--dataset_dir  /root/to/the/raw/DCM/files  \
--dump-root /root/to/the/prepared/files
```

## Generate the list of samples for testing
After running this command, a txt file named val_list.txt will be generated which is used for testing. The txt file that we used during our training process is also provided in this repository.
```
python3 generate_val_set.py  \
--dataset_dir  /root/to/the/prepared/jpg/files 
```

## Training the Baseline model
```
python3 train.py \
--dataset_dir /root/to/the/prepared/files  \
--label_dir /root/to/the/label/csv \
--batch-size 4  --FCCMnet Baseline  \
--lr 1e-4    --epochs 100  --name Baseline
```

## Training Image-Wise and Patch-Wise models
```
python3 train.py \
--dataset_dir /root/to/the/prepared/files  \
--label_dir /root/to/the/label/csv \
--batch-size 4  --FCCMnet PatchWise  \
--lr 1e-4    --epochs 100  --name PatchWiseNetwork
```

```
python3 train.py \
--dataset_dir /root/to/the/prepared/files  \
--label_dir /root/to/the/label/csv \
--batch-size 4  --FCCMnet ImageWise  \
--lr 1e-4    --epochs 100  --name ImageWiseNetwork
```

## Training ResNet model
```
python3 train.py \
--dataset_dir /root/to/the/prepared/files  \
--label_dir /root/to/the/label/csv \
--batch-size 4  --FCCMnet ResNet_FCCM  \
--lr 1e-4    --epochs 100  --name ResNet18
```

## Training CatNet model
```
python3 train.py \
--dataset_dir /root/to/the/prepared/files  \
--label_dir /root/to/the/label/csv \
--batch-size 4  --FCCMnet CatNet_FCCM  \
--lr 1e-4    --epochs 100  --name CatNet18
```

##  Visualization during training process
run the command under the checkpoints folder:
```
tensorboard --logdir=./
```
and visualize the training progress by opening https://localhost:6006 on your browser.

