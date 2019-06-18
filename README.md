# YOLO
Implementation of [YOLO](https://arxiv.org/abs/1506.02640).

## Requirements
* Python3
* numpy
* opencv
* tensorflow

## Before training
```
$ ./prepare.sh
```

## Training
```
$ python train.py {directory to load training and test data} {directory to save model} {epoch size} {learning rate} {start epoch}
```

### Data directory structure
```
{data directory to load training and test data}
    ├── JPEGImages
    └── Segmentations
```

## Prediction
Prediction result is normalized between 0 and 1.
```
$ python predictor.py {image file} {directory model is saved}
```

### Changes from paper
* Batch size is 16.
* Optimizer is a simple gradient discent optimizer.
