# UniDistill: A Universal Cross-Modality Knowledge Distillation Framework for 3D Object Detection in Bird's-Eye View

This is the official implementation of ***[UniDistill](https://arxiv.org/abs/2303.15083)*** **(CVPR2023 highlight✨, 10% of accepted papers)**. UniDistill offers a universal cross-modality knowledge distillation framework for different teacher and student modality combinations. The core idea is aligning the intermediate BEV features and response features that are produced by all BEV detectors.

<center>
<img src='figs/unidistil_framework.png'>
</center>


## Getting Started
### Installation
**Step 0.** Install [pytorch](https://pytorch.org/)(v1.9.0).

**Step 1.** Install [MMCV-full==1.4.2](https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html), MMDetection2D==2.20.2, [MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

**Step 2.** Install requirements.
```shell
pip install -r requirements.txt
```
**Step 3.** Install UniDistill(gpu required).
```shell
python setup.py develop
```

### Data preparation
**Step 0.** Download [nuScenes official dataset](https://www.nuscenes.org/nuscenes).

**Step 1.** Create a folder `/data/dataset/` and put the dataset in it.

The directory will be as follows.
```
├── data
│   ├── dataset
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```
**Step 2.** Download the [infos](https://drive.google.com/file/d/1UZzSSQiD3mr3Zsgj3MS-Hg5tix3-2ZKP/view?usp=share_link) and put them in `/data/dataset/`
The directory will be as follows.
```
├── data
│   ├── dataset
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
|   |   ├── nuscenes_test_meta.pkl
|   |   ├── nuscenes_v1.0-trainval_meta.pkl
|   |   ├── test_info.pkl
|   |   ├── train_info.pkl
|   |   ├── val_info.pkl
```

### Testing
**Step 0.** Download the [checkpoint models](https://drive.google.com/file/d/1TNqjJoqUhYP2_qZncPStF1mnRV4__sUB/view?usp=share_link)

**Step 1.**  Generate the result
If the modality of checkpoint is camera, run the following command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/BEVFusion_nuscenes_centerhead_camera_exp.py -d 0-3 -b 1 -e 20 --sync_bn 1 --no-clearml --infer  --ckpt <PATH_TO_CHECKPOINT>
```
If the modality of checkpoint is LiDAR, change the command as follow:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/BEVFusion_nuscenes_centerhead_camera_exp.py -d 0-3 -b 1 -e 20 --sync_bn 1 --no-clearml --infer  --ckpt <PATH_TO_CHECKPOINT>
```
**Step 2.**  Upload the result to the [evaluation server](https://eval.ai/web/challenges/challenge-page/356/)
The result named "nuscenes_results.json" is in the folder "nuscenes" in the parent folder of the tested checkpoint.
### Evaluation
**Step 0.** Download the checkpoint models as in "Testing"

**Step 1.**  Generate the result
If the modality of checkpoint is camera, run the following command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/BEVFusion_nuscenes_centerhead_camera_exp.py -d 0-3 -b 1 -e 20 --sync_bn 1 --no-clearml --eval  --ckpt <PATH_TO_CHECKPOINT>
```
If the modality of checkpoint is LiDAR, change the command as follow:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/BEVFusion_nuscenes_centerhead_camera_exp.py -d 0-3 -b 1 -e 20 --sync_bn 1 --no-clearml --eval  --ckpt <PATH_TO_CHECKPOINT>
```
### Training
**Step 0.** Train the teacher
Training of the detector of one <MODALITY>:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/BEVFusion_nuscenes_centerhead_<MODALITY>_exp.py -d 0-3 -b 1 -e 20 --sync_bn 1 --no-clearml
```
**Step 1.**  Train the student
Put the checkpoint of the teachers to `unidistill/exps/multisensor_fusion/BEVFusion/tmp/`. Train the teacher of <MODALITY_1> and student of <MODALITY_2>
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/BEVFusion_nuscenes_centerhead_<MODALITY_2>_exp_distill_<MODALITY_1>.py -d 0-3 -b 1 -e 20 --sync_bn 1 --no-clearml
```
## Citation
If you find this project useful in your research, please consider citing:

```
@inproceedings{zhou2023unidistill,
  title={UniDistill: A Universal Cross-Modality Knowledge Distillation Framework for 3D Object Detection in Bird’s-Eye View},
  author={Shengchao Zhou and Weizhou Liu and Chen Hu and Shuchang Zhou and Chao Ma},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
