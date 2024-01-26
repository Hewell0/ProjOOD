# Out-of-distribution detection based on subspace projection of high-dimensional features output by the last convolutional layer 


This is a [PyTorch](http://pytorch.org) implementation for detecting out-of-distribution examples in neural networks. The method is described in the paper Out-of-distribution detection based on subspace projection of high-dimensional features output by the last convolutional layer by Qiuyu Zhu,Yiwei He.


## Running the code

### Dependencies

* CUDA 8.0
* PyTorch
* Anaconda2 or 3

### Downloading  Out-of-Distribtion Datasets
We provide download links of five out-of-distributin datasets:

* **[Tiny-ImageNet (crop)](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)**
* **[Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)**
* **[LSUN (crop)](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)**
* **[LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)**
* **[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)**

Here is an example code of downloading Tiny-ImageNet (crop) dataset. In the **root** directory, run

```
mkdir data
cd data
wget https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
tar -xvzf Imagenet.tar.gz
cd ..
```


### Running

1. Run train.py  to train the ID data classfier.
2. Run OOD_test.py to get the immediate metrics.
3. Run svm.py  to fuse the metrics.
4. Run metrics.py  to get the experimental results.
5. densnet.py is the DENSNET-100 network, which contains PEDCC layer
6. center_pedcc.py to generate the PEDCC points
7. conf.config is config document.
8. utils.py is training progress code。



### License
Please refer to the [LICENSE]([ProjOOD/LICENSE at main · Hewell0/ProjOOD (github.com)](https://github.com/Hewell0/ProjOOD/blob/main/LICENSE) ).