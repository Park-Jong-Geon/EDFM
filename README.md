# Ensemble Distribution Distillation via Flow Matching

Official implementation of _Ensemble Distribution Distillation via Flow Matching_ [(`ICML 2025`)(https://icml.cc/virtual/2025/poster/43616)].



## Installation
#### 1. Google Cloud TPUs (https://cloud.google.com/tpu)

Clone this repository.
```bash
git clone https://github.com/Park-Jong-Geon/EDFM.git
```

As JAX 0.4.3 is now deprecated, please download `jaxlib-0.4.3-cp39-cp39-manylinux2014_x86_64.whl` from `https://dashboard.stablebuild.com/pypi-deleted-packages/pkg/jaxlib/0.4.3`.
Move the wheel file into the cloned repository.

Create a new conda environment.
```bash
conda env create -f edfm.yaml
```

#### 2. CUDA

For installations in CUDA environments, you only need to modify `jaxlib` version accordingly in `edfm.yaml`


## Download datasets
### CIFAR-10
```bash
cd data
wget https://www.dropbox.com/s/8s5unpaxdrt6r7e/CIFAR10_HMC.tar.gz
tar -xvzf CIFAR10_HMC.tar.gz
mv CIFAR10_HMC CIFAR10_x32
cd -
```
### CIFAR-100
```bash
cd data
wget https://www.dropbox.com/s/bvljrqsttemfdzv/CIFAR100_HMC.tar.gz
tar -xvzf CIFAR100_HMC.tar.gz
mv CIFAR100_HMC CIFAR100_x32
cd -
```


## Train MultiSWAG teachers
To train MultiSWAG teachers for CIFAR-10, execute the following command. (Modify the config file accordingly for CIFAR-100.)
Note that pretrained teachers for CIFAR-10 and CIFAR-100 datasets are provided in `checkpoints_teacher` directory.
```bash
python sgd_swag.py\ 
    --config config/teacher/c10_frnrelu_sgd_swag.yaml \ 
    --save {save_dir} \ 
    --exp_name {exp_name}
```

## EDFM
To train EDFM for CIFAR-10, execute the following command. (Modify the config file accordingly for CIFAR-100.)
```bash
python edfm.py\ 
    --config config/distillation/edfm_c10_frnrelu.yaml\ 
    --save {save_dir}\ 
    --exp_name {exp_name}
```

## Baselines
### KD (https://arxiv.org/abs/1503.02531)
To train KD for CIFAR-10, execute the following command. (Modify the config file accordingly for CIFAR-100.)
```bash
python kd_and_endd.py\ 
    --config config/distillation/kd_c10_frnrelu.yaml\ 
    --save {save_dir}\ 
    --exp_name {exp_name}
```

### EnDD (https://arxiv.org/abs/2105.06987)
To train EnDD for CIFAR-10, execute the following command. (Modify the config file accordingly for CIFAR-100.)
```bash
python kd_and_endd.py\ 
    --config config/distillation/endd_c10_frnrelu.yaml\ 
    --save {save_dir}\ 
    --exp_name {exp_name}
```

### FED (https://arxiv.org/abs/2206.02183)
To train FED for CIFAR-10, execute the following command. (Modify the config file accordingly for CIFAR-100.)
```bash
python fed.py\ 
    --config config/distillation/fed_c10_frnrelu.yaml\ 
    --save {save_dir}\ 
    --exp_name {exp_name}
```

## Citation
Please consider citing our paper if you found it useful. 
```
@inproceedings{park2025edfm,
    title     = {Ensemble Distribution Distillation via Flow Matching},
    author    = {Park, Jonggeon and Nam, Giung and Kim, Hyunsu and Yoon, Jongmin and Lee, Juho},
    booktitle = {International Conference on Machine Learning (ICML)},
    year      = {2025},
}
```