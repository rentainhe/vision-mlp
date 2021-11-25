# vision-mlp
A collection of SOTA vision mlp models based on pytorch, this repo is mainly built upon [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), thanks a lot for their great help and also feel truly grateful for [Jittor-MLP](https://github.com/liuruiyang98/Jittor-MLP).

## Updates
- (2021.11.22) Release vision-mlp repo with [g-mlp](/models/g_mlp.py), [mlp-mixer](/models/mlp_mixer.py), [res-mlp](/models/res_mlp.py), [swin-mlp](/models/swin_mlp.py) models


## Supported models
- [x] [Mlp-Mixer](/configs/mlp-mixer)
- [x] [ResMLP](/configs/res-mlp)
- [x] [gMLP](/configs/g-mlp)
- [ ] [Vision Permutator]()
- [x] [S2-MLP](/configs/s2-mlp)
- [ ] [S2-MLP-V2]()
- [ ] [ConvMLP]()
- [ ] [RaftMLP]()
- [ ] [Sparse MLP]()
- [ ] [Hire-MLP]()
- [ ] [GFNet]()
- [ ] [CycleMLP]()


## Usage

<details>
<summary> <b> Installation </b> </summary>

- Clone this repo:

```bash
git clone https://github.com/rentainhe/vision-mlp.git
cd vision-mlp
```

- Create a conda virtual environment and activate it:

```bash
conda create -n mlp python=3.7 -y
conda activate mlp
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install `cupy`:
```bash
pip install cupy-cuda101
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 einops==0.3.2
```

</details>

<details>
<summary> <b> Data preparation </b> </summary>

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```
</details>

<details>
<summary> <b> Evaluation </b> </summary>

### Evaluation

To evaluate a pre-trained `ResMLP` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

For example, to evaluate the `res_mlp_12` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/res-mlp/res_mlp_12.yaml --resume res_mlp_12.pth --data-path <imagenet-path>
```
</details>

<details>
<summary> <b> Training from scratch </b> </summary>

To train a `ResMLP` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

**Notes**:

- To use zipped ImageNet instead of folder dataset, add `--zip` to the parameters.
    - To cache the dataset in the memory instead of reading from files every time, add `--cache-mode part`, which will
      shard the dataset into non-overlapping pieces for different GPUs and only load the corresponding one for each GPU.
- When GPU memory is not enough, you can try the following suggestions:
    - Use gradient accumulation by adding `--accumulation-steps <steps>`, set appropriate `<steps>` according to your need.
    - Use gradient checkpointing by adding `--use-checkpoint`, e.g., it saves about 60% memory when training `Swin-B`.
      Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.
    - We recommend using multi-node with more GPUs for training very large models, a tutorial can be found
      in [this page](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
- To change config options in general, you can use `--opts KEY1 VALUE1 KEY2 VALUE2`, e.g.,
  `--opts TRAIN.EPOCHS 100 TRAIN.WARMUP_EPOCHS 5` will change total epochs to 100 and warm-up epochs to 5.
- For additional options, see [config](config.py) and run `python main.py --help` to get detailed message.

For example, to train `ResMLP` with 8 GPU on a single node for 300 epochs, run:

`ResMLP-12`:
```python
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/res-mlp/res_mlp_12.yaml --data-path <imagenet-path> --batch-size 128 
```


</details>


<details>
<summary> <b> Throughput </b> </summary>

To measure the throughput, run:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 64 --throughput --amp-opt-level O0
```
</details>


## Citations

```bibtex
@misc{tolstikhin2021mlpmixer,
    title   = {MLP-Mixer: An all-MLP Architecture for Vision},
    author  = {Ilya Tolstikhin and Neil Houlsby and Alexander Kolesnikov and Lucas Beyer and Xiaohua Zhai and Thomas Unterthiner and Jessica Yung and Daniel Keysers and Jakob Uszkoreit and Mario Lucic and Alexey Dosovitskiy},
    year    = {2021},
    eprint  = {2105.01601},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{hou2021vision,
    title   = {Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition},
    author  = {Qibin Hou and Zihang Jiang and Li Yuan and Ming-Ming Cheng and Shuicheng Yan and Jiashi Feng},
    year    = {2021},
    eprint  = {2106.12368},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{liu2021pay,
  title={Pay Attention to MLPs},
  author={Liu, Hanxiao and Dai, Zihang and So, David R and Le, Quoc V},
  journal={arXiv preprint arXiv:2105.08050},
  year={2021}
}
```

```bibtex
@article{touvron2021resmlp,
  title={Resmlp: Feedforward networks for image classification with data-efficient training},
  author={Touvron, Hugo and Bojanowski, Piotr and Caron, Mathilde and Cord, Matthieu and El-Nouby, Alaaeldin and Grave, Edouard and Joulin, Armand and Synnaeve, Gabriel and Verbeek, Jakob and J{\'e}gou, Herv{\'e}},
  journal={arXiv preprint arXiv:2105.03404},
  year={2021}
}
```

```bibtex
@article{yu2021s,
  title={S $\^{} 2$-MLPv2: Improved Spatial-Shift MLP Architecture for Vision},
  author={Yu, Tan and Li, Xu and Cai, Yunfeng and Sun, Mingming and Li, Ping},
  journal={arXiv preprint arXiv:2108.01072},
  year={2021}
}
```

```bibtex
@article{li2021convmlp,
  title={ConvMLP: Hierarchical Convolutional MLPs for Vision},
  author={Li, Jiachen and Hassani, Ali and Walton, Steven and Shi, Humphrey},
  journal={arXiv preprint arXiv:2109.04454},
  year={2021}
}
```

```bibtex
@article{tatsunami2021raftmlp,
  title={RaftMLP: Do MLP-based Models Dream of Winning Over Computer Vision?},
  author={Tatsunami, Yuki and Taki, Masato},
  journal={arXiv preprint arXiv:2108.04384},
  year={2021}
}
```

```bibtex
@article{tang2021sparse,
  title={Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?},
  author={Tang, Chuanxin and Zhao, Yucheng and Wang, Guangting and Luo, Chong and Xie, Wenxuan and Zeng, Wenjun},
  journal={arXiv preprint arXiv:2109.05422},
  year={2021}
}
```

```bibtex
@article{guo2021hire,
  title={Hire-MLP: Vision MLP via Hierarchical Rearrangement},
  author={Guo, Jianyuan and Tang, Yehui and Han, Kai and Chen, Xinghao and Wu, Han and Xu, Chao and Xu, Chang and Wang, Yunhe},
  journal={arXiv preprint arXiv:2108.13341},
  year={2021}
}
```

```bibtex
@article{rao2021global,
  title={Global filter networks for image classification},
  author={Rao, Yongming and Zhao, Wenliang and Zhu, Zheng and Lu, Jiwen and Zhou, Jie},
  journal={arXiv preprint arXiv:2107.00645},
  year={2021}
}
```

```bibtex
@article{chen2021cyclemlp,
  title={Cyclemlp: A mlp-like architecture for dense prediction},
  author={Chen, Shoufa and Xie, Enze and Ge, Chongjian and Liang, Ding and Luo, Ping},
  journal={arXiv preprint arXiv:2107.10224},
  year={2021}
}
```

```bibtex
@article{lian2021mlp,
  title={As-mlp: An axial shifted mlp architecture for vision},
  author={Lian, Dongze and Yu, Zehao and Sun, Xing and Gao, Shenghua},
  journal={arXiv preprint arXiv:2107.08391},
  year={2021}
}
```