# YOLO
Implementation of [YOLO](https://arxiv.org/abs/1506.02640).

## Requirements
* Python3
* numpy
* tensorflow
* Pillow
* Before running python code, download and uncompress [training data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)

## Training
```
$ python train.py {directory to load training and test data} {directory to save model}
```

### Data directory structure
```
{data directory to load training and test data}
    ├── Annotations
    └── JPEGImages
```

### Changes from paper
* Training and validation data is VOC2007.
* Use sigmoid activation function at last layer.
* initial value is random number according to normal distribution that mean is 0 and standard deviation is 0.03.
* Weight decay coefficient is 0.00005.
* Batch size is 16.
* Optimizer is a simple gradient discent optimizer.
