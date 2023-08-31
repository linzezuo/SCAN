# SCAN
## Codes for paper "Spatial and Channel Aggregation Network for lightweight Image Super-Resolution".
## Environment
PyTorch >= 1.8
BasicSR >= 1.3.5
## Training and Testing
### Training with the example option:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 train.py -opt options/trian_SCAN.yml --launcher pytorch
### Testing with the example option:
python test.py -opt options/SCAN_light.yml













