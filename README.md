[![arXiv](https://img.shields.io/badge/arXiv-2204.09138-b31b1b.svg)](https://arxiv.org/abs/2204.09138)
![code visitors](https://visitor-badge.glitch.me/badge?page_id=vLAR-group/RangeUDF)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/vLAR-group/RangeUDF/blob/main/LICENSE)
[![Twitter Follow](https://img.shields.io/twitter/follow/vLAR_Group?style=social)](https://twitter.com/vLAR_Group)

# RangeUDF: Semantic Surface Reconstruction from 3D Point Clouds

This is the official repository of the **RangeUDF**. For technical details, please refer to:

**RangeUDF: Semantic Surface Reconstruction from 3D Point Clouds** <br />
[Bing Wang](https://www.cs.ox.ac.uk/people/bing.wang/), [Zhengdi Yu](), [Bo Yang<sup>*</sup>](https://yang7879.github.io/), [Jie Qin](https://sites.google.com/site/firmamentqj/), [Toby Breckon](https://breckon.org/toby/), [Ling Shao](https://scholar.google.com/citations?user=z84rLjoAAAAJ&hl=en), [Niki Trigoni](https://www.cs.ox.ac.uk/people/niki.trigoni/), [Andrew Markham](https://www.cs.ox.ac.uk/people/andrew.markham/) <br />
**[[Paper](http://arxiv.org/abs/2204.09138)] [[Video](https://youtu.be/YahEnX1z-yw)]** <br />

### Video Demo (Youtube)
<p align="center"> <a href="https://youtu.be/YahEnX1z-yw"><img src="./figs/RangeUDF.jpg" width="80%"></a> </p>

### Qualitative Results
| ![2](./figs/fig_scannet_scene0015_rec.gif)   | ![z](./figs/fig_scannet_scene0015_sem.gif) |
| --------------------------------------- | ------------------------------------- |
| ![2](./figs/fig_scannet_scene0221_rec.gif)   | ![z](./figs/fig_scannet_scene0221_sem.gif) |
| ![2](./figs/fig_scannet_scene0500_rec.gif)   | ![z](./figs/fig_scannet_scene0500_sem.gif) |

## 1. Installation

RangeUDF uses a Conda environment that makes it easy to install all dependencies.

Create the `rangeudf` Conda environment (Python 3.7) with [miniconda](https://docs.conda.io/en/latest/miniconda.html) and install all dependencies. 

```bash
conda env create -f environment.yml
```
***Note:*** You can install extensions with following command:

```bash
cd tools/cpp_wrappers
bash compile_wrappers.sh
cd ../nearest_neighbors
python setup.py install --inplace
```
### 1.1 Datasets

In this paper, we consider the following four different datasets:

#### (1) [Synthetic Rooms](https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/data/room_watertight_mesh.zip)

This is a synthetic indoor dataset, consisting of 5000 scenes (3,750 for training, 250 for validation and 1,000 for testing). Each scene has several objects (chair, sofa, lamp, cabinet, table) from [ShapeNet](https://shapenet.org/) . We follow the same split setting in [ConvOCC](https://github.com/autonomousvision/convolutional_occupancy_networks) and use the whole test set to conduct quantitative evaluation. 

#### (2) [SceneNN](https://drive.google.com/file/d/1d_ILfaxpJBpiiwCZtvC4jEKnixEr9N2l/view?usp=sharing)

This is an RGB-D dataset with 76 indoor scenes for the task of 3D semantic segmentation. There are 56 scenes for training and 20 scenes for testing as [Pointwise](https://github.com/hkust-vgd/pointwise) with 11 semantic classes. We adopt the same split setting in our experiments.
#### (3) [ScanNet](http://www.scan-net.org/)

ScanNet contains 1,513 real-world rooms collected by an RGB-D camera. There are 20 semantic classes in the evaluation of 3D semantic segmentation. In particular, there are 1,201 scans for training and 312 for validation. Since ScanNet does not provide an online benchmark for surface reconstruction, we use the validation as our testing set and directly sample surface points from the provided raw (without alignment) non-watertight meshes.
#### (4) [2D-3D-S](https://github.com/alexsax/2D-3D-Semantics)

It covers 6 large-scale indoor areas with 271 rooms (Area-1: 44, Area-2: 40, Area-3: 23, Area4: 49, Area-5: 67, Area-6: 48) captured by Matterport sensors. There are 13 annotated semantic classes for this dataset. A non-watertight mesh is provided for each room. Note that, Area-5 is split into Area-5a and Area-5b, in which 47 rooms are unevenly broken into two parts. To avoid the imbalanced data introduced by Area-5, we choose Area-1∼ Area-4 as our training set and Area-6 as the testing set.
### 1.2 Data preprocessing

The links to the four datasets above point to the original datasets, which you can download and unzip into the corresponding `/data/dataset_name/source/` and then preprocess the data with the following commands:

```bash
cd data/dataset_name/
bash preprocess.sh
cd ../
python labeled.py
```
Note: The outermost folder name of the dataset does not need to be preserved. For example, `synthetic_room_dataset/scenes5/room0*` should be translated to `data/synthetic/source/room0*`.

Naturally, we also provide already processed datasets for download.
## 2. Training

For the training of our standard RangeUDF , you can simply run the following command with a chosen config file specifying data directory and hyper-params.

```bash

CUDA_VISIBLE_DEVICES=0 python train_s3dis.py --config  configs/train/s3dis.txt --concat 3 --reg_coef 0.1 --log_dir rec_no_schedule  --in_dim 3 --num_points 10000 

```
Other working modes and set-ups can be also made via the above command by choosing different config files.

***Note:*** Synthetic Rooms has no semantic labels. The task in the config file provides three types of inputs: `rec, sem, and joint`, which correspond to `Reconstruction, Semantic Segmentation, Semantic Segmentation and Reconstruction`, respectively.

## 3. Evaluation

In this paper, we use reconstruction metrics (CD-L<sub>1</sub>, CD-L<sub>2</sub>, F-score) for evaluation, and semantic segmentation metrics (mIoU, OA). 

### (1) Reconstruction

#### Mesh Generation

For mesh generation, you can change the config file and then run:

```bash
CUDA_VISIBLE_DEVICES=0 python gen_s3dis.py --config configs/test/s3dis.txt --concat 3 --reg_coef 0.1 --log_dir rec_no_schedule  --in_dim 3 --num_points 10000 --ckpt ckpt_name
```

***Note:*** Checkpoints are saved by default in `experiments/exp_name/log_dir/checkpoints/`. By default one is `checkpoint_latest.tar` and one is `checkpoint_best.tar`. You can set any `ckpt_name` you want, it's just a flag. If you enter `-ckpt` then `best` is selected, otherwise `latest` is selected. We save the test setup as a pkl file to make it easier for you to generate with different setups, such as `num_points`.
 
#### Evaluation

For evaluation, you can change the config file and then run:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/test/s3dis.txt --log_dir rec_no_schedule --ckpt ckpt_name
```
***Note:*** All datasets use the same `evaluate.py`.

### (2) Semantic Segmentation

Waiting for update

### (3) How to use pretrain model
We provide pre-trained models using the default settings file. You can just place the file in the corresponding dir `experiments/{exp_name}/{log_dir}/checkpoints/`, then rename as `checkpoint_best.tar`.
```bash
CUDA_VISIBLE_DEVICES=0 python gen_{exp_name}.py --config configs/test/{exp_name}.txt --concat 3  --log_dir {log_dir}  --in_dim 3 --num_points 10000 --ckpt {any ckpt_name you like}

CUDA_VISIBLE_DEVICES=0 python evaluate.py --config configs/test/{exp_name}.txt --log_dir {log_dir} --ckpt {ckpt_name same as up line}

### Citation
If you find our work useful in your research, please consider citing:

      @article{wang2022rangeudf,
      title={RangeUDF: Semantic Surface Reconstruction from 3D Point Clouds},
      author={Bing, Wang and Zhengdi, Yu and Yang, Bo and Jie, Qin and Toby, Breckon and Ling, Shao and Trigoni, Niki and Markham, Andrew},
      journal={arXiv preprint arXiv:2204.09138},
      year={2022}
    }
