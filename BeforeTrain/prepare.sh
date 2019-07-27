#!/bin/sh

curl -OsS http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
curl -OsS http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar -C VOCdevkit/
mv VOCdevkit/VOCdevkit/VOC2012 VOCdevkit/VOC2012
python3 get_voc.py
rm -rf VOCdevkit/VOCdevkit/ VOCdevkit/VOC2012/ VOCdevkit/VOC2007/ VOCdevkit/VOC2007+2012/Annotations/ VOCtrainval_06-Nov-2007.tar VOCtrainval_11-May-2012.tar

curl -OsS http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz
tar xfz imagenet_fall11_urls.tgz
curl -OsS http://image-net.org/archive/words.txt
mkdir ImageNet
python3 get_imagenet.py
rm imagenet_fall11_urls.tgz
