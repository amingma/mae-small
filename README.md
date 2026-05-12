# MAE Small

Light implementation of [Masked Autoencoders are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377) paper

## Directory Structure

```
mae-small/
├── images/                    # Sample images from ImageNet-1K
├── logs/                      # Training logs
├── .gitignore
├── mae_imagenet.py            # Pretraining script for MAE
├── vit_base.py                # Training script for ViT-Base baseline 
├── vit_base_mae.py            # Training script for ViT-Base with MAE weights preloaded
├── mae_visualize_imagenet.py  # Image reconstruction script
├── train_imagenet.sh          # Shell script to submit job on Delta cluster
├── save_imagenet.sh    
└── README.md

```

## Usage

### Load Imagenet-1K dataset

Adjust the output directory in save_imagenet.py as appropiate, and run the save script. This will take some time, so I recommend running the script in a tmux session.

```
sbatch save_imagenet.sh
```

### Train models

Make sure to adjust the file in the shell script to reflect the training script you would like to run. Also, adjust the directories in the training scripts to be the same as where you saved them locally.

```
sbatch train_imagenet.sh
```

### Image reconstruction

After training the masked autoencoder, adjust the save path in the reconstruction script to point towards the .pt file where the weights were saved. Feel free to add your own images to the images folder!

```
python mae_visualize_imagenet.py --images_dir images --n_images 3
```
