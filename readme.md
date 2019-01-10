# Domain Transfer with CycleGAN

Use CycleGAN to transform **gray MNIST** to **colorful MNIST**.



## Prepare Data

1. Download `mnist.npz` from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz, put it into `data/mnist`.
2. run `gen_color_mnist.py`.

```python
    python gen_color_mnist.py
```

## Train and Transform

1. run `cyclegan.py`.

```python
    python cyclegan.py
```

---

## Transformed Images

input | Epoch 0 | Epoch 40 | Epoch 99
:---: | :---: | :---:  | :---:
<img src="figs/cyclegan/ori_A.png" width=280px> | <img src="figs/cyclegan/AtoB_epoch_0.png" width=280px> | <img src="figs/cyclegan/AtoB_epoch_40.png" width=280px> | <img src="figs/cyclegan/AtoB_epoch_99.png" width=280px>
<img src="figs/cyclegan/ori_B.png" width=280px> | <img src="figs/cyclegan/BtoA_epoch_0.png" width=280px> | <img src="figs/cyclegan/BtoA_epoch_40.png" width=280px> | <img src="figs/cyclegan/BtoA_epoch_99.png" width=280px>

## Loss plots

reconstruct loss | G&D loss A | G&D loss B |
:---: | :---: | :---:
<img src="figs/cyclegan/reconstruct_loss.png" width=400px> | <img src="figs/cyclegan/GD_loss_A.png" width=400px> | <img src="figs/cyclegan/GD_loss_B.png" width=400px>



## Reference

1. [Understanding and Implementing CycleGAN in TensorFlow](https://hardikbansal.github.io/CycleGANBlog/)
2. [CycleGAN-tensorflow](https://github.com/hardikbansal/CycleGAN/)