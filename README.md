<p align="center"><img width="40%" src="jpg/logo.jpg" /></p>

--------------------------------------------------------------------------------
This repository provides a PyTorch implementation of [StarGAN](https://arxiv.org/abs/1711.09020). StarGAN can flexibly translate an input image to any desired target domain using only a single generator and a discriminator. The demo video for StarGAN can be found [here](https://www.youtube.com/watch?v=EYjdLppmERE).

<p align="center"><img width="100%" src="jpg/main.jpg" /></p>

<br/>

## Paper
[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) <br/>
[Yunjey Choi](https://github.com/yunjey)<sup> 1,2</sup>, [Minje Choi](https://github.com/mjc92)<sup> 1,2</sup>, [Munyoung Kim](https://www.facebook.com/munyoung.kim.1291)<sup> 2,3</sup>, [Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<sup> 2</sup>, [Sung Kim](https://www.cse.ust.hk/~hunkim/)<sup> 2,4</sup>, and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)<sup> 1,2</sup>    <br/>
<sup>1 </sup>Korea University, <sup>2 </sup>Clova AI Research (NAVER Corp.), <sup>3 </sup>The College of New Jersey, <sup> 4 </sup>HKUST  <br/>
IEEE Conference on Computer Vision and Pattern Recognition ([CVPR](http://cvpr2018.thecvf.com/)), 2018 (<b>Oral</b>) 

<br/>

## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)


<br/>

## Usage

### 1. Cloning the repository
```bash
$ git clone https://github.com/liangxiaoyun/Facial-Attribute-Transfer
```

### 2. Downloading the dataset
To download the CelebA dataset:
```bash
$ bash download.sh celeba
```

To download the RaFD dataset, you must request access to the dataset from [the Radboud Faces Database website](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main). Then, you need to create a folder structure as described [here](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md).

### 3. Training
To train StarGAN on CelebA, run the training script below. See [here](https://github.com/yunjey/StarGAN/blob/master/jpg/CelebA.md) for a list of selectable attributes in the CelebA dataset. If you change the `selected_attrs` argument, you should also change the `c_dim` argument accordingly.

```bash
$CUDA_VISIBLE_DEVICES=1 python main.py --mode train --dataset CelebA --image_size 128 --c_dim 5 --celeba_image_dir data/CelebA/CelebA_nocrop/images --attr_path data/list_attr_celeba.txt --sample_dir stargan/Celeba/samples --log_dir stargan/Celeba/logs --model_save_dir stargan/Celeba/models --result_dir stargan/Celeba/results --num_iters 300000
```

To train StarGAN on RaFD:

```bash
$CUDA_VISIBLE_DEVICES=1 python main.py --mode train --dataset data/RaFD/train --image_size 128 --c_dim 7 --sample_dir stargan/rafd/samples --log_dir stargan/rafd/logs --model_save_dir stargan/rafd/models --result_dir stargan/rafd/results
```

To train StarGAN on both CelebA and RafD:

```bash
$python main.py --mode=train --dataset Both --image_size 128 --c_dim 5 --c2_dim 7 --sample_dir stargan/both/samples --log_dir stargan/both/logs --model_save_dir stargan/both/models --result_dir stargan/both/results
```


To train StarGAN on your own dataset, create a folder structure in the same format as [RaFD](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md) and run the command:

```bash
$python main.py --mode train --dataset RaFD --image_size 128 --c_dim 7 --rafd_image_dir data/KDEF/train --sample_dir stargan/KDEF/samples --log_dir stargan/KDEF/logs --model_save_dir stargan/KDEF/models --result_dir stargan/KDEF/results
```


### 4. Testing

To test StarGAN on CelebA:

```bash
$python main.py --mode test --dataset CelebA --image_size 128 --c_dim 5 --model_save_dir stargan/Celeba/models --result_dir stargan/Celeba/results --celeba_image_dir data/testimage--selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --celeba_image_dir data/testimage --attr_path data/test_list_attr_celeba.txt ----test_iters 300000
```

To test StarGAN on RaFD:

```bash
$python main.py --mode test --dataset RaFD --image_size 128 --c_dim 7 --model_save_dir stargan/rafd/models --result_dir stargan/rafd/results --rafd_image_dir data/testimage --attr_path data/test_list_attr_celeba.txt
```


To test StarGAN on both CelebA and RaFD:

```bash
$python main.py --mode test --dataset Both --image_size 128 --c_dim 5 --c2_dim 7 --model_save_dir stargan/both/models --result_dir stargan/both/results --rafd_image_dir data/RaFD/test --celeba_image_dir data/testimage --attr_path data/test_list_attr_celeba.txt
```


To test StarGAN on your own dataset:

```bash
$python main.py --mode test --dataset RaFD --image_size 128 --c_dim 7 --model_save_dir stargan/KDEF/models --result_dir stargan/KDEF/results --rafd_image_dir data/testimage --attr_path data/test_list_attr_celeba.txt
```

### 5. Pretrained model
To download a pretrained model checkpoint, run the script below. The pretrained model checkpoint will be downloaded and saved into `stargan_celeba_256/models` directory.

```bash
$ bash download.sh pretrained-celeba-256x256
```

To translate images using the pretrained model, run the evaluation script below. The translated images will be saved into `stargan/stargan_celeba_256/results` directory.

```bash
$ python main.py --mode test --dataset CelebA --image_size 256 --c_dim 5 --model_save_dir='stargan/stargan_celeba_256/models' --result_dir='stargan/stargan_celeba_256/results'
```

### 6. 生成可执行文件
```bash
$pip install pyinstaller
$pyinstaller main.py
```
生成build和dist文件夹，双击，dist/main/main.exe即可打开GUI界面，如GUI_image.png

<br/>

## Results


### 1. Facial Expression Synthesis on RaFD------STARGAN
<p align="center"><img width="100%" src="jpg/result_rafd.png" /></p>

### 2. Facial Expression Synthesis on CelebA--------STARGAN
<p align="center"><img width="100%" src="jpg/result_celeba.jpg" /></p>

### 3. Facial Expression Synthesis on CelebA----SN-STARGAN
<p align="center"><img width="100%" src="jpg/result_celeba_SN.jpg" /></p>

<br/>


