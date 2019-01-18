# Domain Transfer with CycleGAN

Use CycleGAN to transform **gray MNIST** to **colorful MNIST**.



## Prepare Data

1. Download `mnist.npz` from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz, put it into `data/mnist`.
2. run `gen_color_mnist.py`.

```python
    python gen_color_mnist.py
```

## Train and Transform

run `cyclegan.py` to train cyclegan.

```python
    python cyclegan.py
```
run `daegan.py` to train daegan.

```python
    python daegan.py
```

## CycleGAN
### 1. Network Structure

<center>
<img src="figs/networks/cyclegan.png" width=400px>
</center>

### 2. Transformed Images

input | Epoch 0 | Epoch 40 | Epoch 99
:---: | :---: | :---:  | :---:
<img src="figs/cyclegan/ori_A.png" width=280px> | <img src="figs/cyclegan/AtoB_epoch_0.png" width=280px> | <img src="figs/cyclegan/AtoB_epoch_40.png" width=280px> | <img src="figs/cyclegan/AtoB_epoch_99.png" width=280px>
<img src="figs/cyclegan/ori_B.png" width=280px> | <img src="figs/cyclegan/BtoA_epoch_0.png" width=280px> | <img src="figs/cyclegan/BtoA_epoch_40.png" width=280px> | <img src="figs/cyclegan/BtoA_epoch_99.png" width=280px>

### 3. Loss plots

reconstruct loss | G&D loss A | G&D loss B |
:---: | :---: | :---:
<img src="figs/cyclegan/reconstruct_loss.png" width=400px> | <img src="figs/cyclegan/GD_loss_A.png" width=400px> | <img src="figs/cyclegan/GD_loss_B.png" width=400px>

## DaeGAN
### 1. Network Structure

<center>
<img src="figs/networks/daegan.png" width=400px>
</center>

### 2. Transformed Images

input | Epoch 0 | Epoch 10 | Epoch 19
:---: | :---: | :---:  | :---:
<img src="figs/daegan/cycA_19.png" width=280px> | <img src="figs/daegan/cross_AtoB_0.png" width=280px> | <img src="figs/daegan/cross_AtoB_10.png" width=280px> | <img src="figs/daegan/cross_AtoB_19.png" width=280px>
<img src="figs/daegan/cycB_19.png" width=280px> | <img src="figs/daegan/cross_BtoA_0.png" width=280px> | <img src="figs/daegan/cross_BtoA_10.png" width=280px> | <img src="figs/daegan/cross_BtoA_19.png" width=280px>

### 3. AutoEncoder Recovery

input | Epoch 0 | Epoch 10
:---: | :---: | :---:
<img src="figs/daegan/cycA_19.png" width=280px> | <img src="figs/daegan/cycA_0.png" width=280px> | <img src="figs/daegan/cycA_10.png" width=280px>
<img src="figs/daegan/cycB_19.png" width=280px> | <img src="figs/daegan/cycB_0.png" width=280px> | <img src="figs/daegan/cycB_10.png" width=280px> 


### 4. Latent Distribution
Epoch 0 | Epoch 10 | Epoch 19
:---: | :---:  | :---:
<img src="figs/daegan/hist_0.png" width=280px> | <img src="figs/daegan/hist_10.png" width=280px> | <img src="figs/daegan/hist_19.png" width=280px>

## Reference

1. [Understanding and Implementing CycleGAN in TensorFlow](https://hardikbansal.github.io/CycleGANBlog/)
2. [CycleGAN-tensorflow](https://github.com/hardikbansal/CycleGAN/)