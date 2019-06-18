#!/bin/sh

curl -O http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
curl -O http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_11-May-2012.tar -C VOCdevkit/
mv VOCdevkit/VOCdevkit/VOC2012 VOCdevkit/VOC2012
python3 prepare.py
rm -rf VOCdevkit/VOCdevkit/ VOCdevkit/VOC2012/ VOCdevkit/VOC2007/ VOCdevkit/VOC2007+2012/Annotations/ VOCtrainval_06-Nov-2007.tar VOCtrainval_11-May-2012.tar
