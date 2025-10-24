
#!/bin/bash
# Example: train on SOLID (ensure paths are correct)
python3 train.py --data_root /path/to/SOLID --root_vmamba /path/to --ckpt_dir ./checkpoints --epochs 10 --batch_size 4 --pretrained_resnet
# Example: test using best checkpoint
python3 test.py --data_root /path/to/SOLID --ckpt ./checkpoints/tffn_best.pth --root_vmamba /path/to --pretrained_resnet
