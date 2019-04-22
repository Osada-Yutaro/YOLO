# YOLO
Implementation of [YOLO](https://arxiv.org/abs/1506.02640).

Before running python code, download and uncompress [training data](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)

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
