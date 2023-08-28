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
## Results
### Model performance and complexity comparison between our proposed SCAN model and other lightweight SISR methods on Set5 for ×4 SR. The circle sizes indicate the floating-point operations (FLOPs). Our proposed SCAN achieves a good trade-off between model performance and complexity:
![image](https://github.com/linzezuo/SCAN/assets/135433947/ca324bde-fa95-4518-aea1-80230fd0cb5e)
### Comparison of local attribution maps (LAMs) between our SCAN and other efficient light-weight SR models. The LAMs demonstrate the significance of every pixel in the input LR image with respect to the patch marked with a red box's SR. Additionally, we show the contribution area in the third row. It is evident that our SCAN can aggregate more information:
![image](https://github.com/linzezuo/SCAN/assets/135433947/8ea4ade6-e110-42cd-a179-dc6fd827f099)
### Visual comparison about image SR (×4) using SCAN-tiny model in some challenging cases:
![image](https://github.com/linzezuo/SCAN/assets/135433947/c989b2f2-665c-44e6-9c28-63c588db7240)
### Visual comparison about image SR (×4) using SCAN-light model in some challenging cases:
![image](https://github.com/linzezuo/SCAN/assets/135433947/04e4cad3-7050-4dc3-8676-9dc5719d392b)
### Visual comparison about image SR (×4) using SCAN-light model in some challenging remote sensing cases:
![image](https://github.com/linzezuo/SCAN/assets/135433947/1a6b6869-4ae3-4b62-8242-659a4feb0c25)












