import os
import shutil
import xml.etree.ElementTree as ET
import glob
import random
import numpy as np
import yolo

S = yolo.S
B = yolo.B
C = yolo.C
INDICES = yolo.INDICES

def convert_Y(parsed_xml):
    width = float(parsed_xml.find('size/width').text)
    height = float(parsed_xml.find('size/height').text)
    y_data = np.zeros((S, S, 5*B + C), dtype='float32')
    for sx in range(S):
        for sy in range(S):
            b = 0
            for obj in parsed_xml.findall('object'):
                if b == B:
                    break

                xmin = float(obj.find('bndbox/xmin').text)
                ymin = float(obj.find('bndbox/ymin').text)
                xmax = float(obj.find('bndbox/xmax').text)
                ymax = float(obj.find('bndbox/ymax').text)
                xmid = (xmin + xmax)/2
                ymid = (ymin + ymax)/2

                if sx*width/S <= xmid < (sx + 1)*width/S and sy*height/S <= ymid < (sy + 1)*height/S:
                    y_data[sx, sy, C + 5*b] = S*xmid/width - sx
                    y_data[sx, sy, C + 5*b + 1] = S*ymid/height - sy
                    y_data[sx, sy, C + 5*b + 2] = (xmax - xmin)/width
                    y_data[sx, sy, C + 5*b + 3] = (ymax - ymin)/height
                    y_data[sx, sy, C + 5*b + 4] = 1.

                    y_data[sx, sy, INDICES[obj.find('name').text]] = 1.
                    b += 1
    return y_data

def load_dataset(directory):
    if directory[-1] != '/':
        directory = directory + '/'

    os.mkdir(directory + 'Segmentations')

    fs = list(map(lambda y: os.path.splitext(y)[0], list(filter(lambda x: x != '.DS_Store', os.listdir(directory + 'Annotations/')))))
    train = open(directory + 'train.txt', 'w')
    valid = open(directory + 'validation.txt', 'w')

    random.shuffle(fs)
    count = 0
    for f in fs:
        xml = ET.parse(directory + 'Annotations/' + f + '.xml')
        oup = convert_Y(xml)
        np.save(directory + 'Segmentations/' + f + '.npy', oup)

        if count%4 == 0:
            valid.write(f + '\n')
        else:
            train.write(f + '\n')
        count += 1

    train.close()
    valid.close()

def join_2007_2012(directory, dir_2007, dir_2012):
    if directory[-1] != '/':
        directory = directory + '/'
    if dir_2007[-1] != '/':
        dir_2007 = dir_2007 + '/'
    if dir_2012[-1] != '/':
        dir_2012 = dir_2012 + '/'

    VOC2007_2012 = 'VOC2007+2012/'
    ANNOTATIONS = 'Annotations/'
    JPEGIMAGES = 'JPEGImages/'

    os.mkdir(directory + VOC2007_2012)
    os.mkdir(directory + VOC2007_2012 + ANNOTATIONS)
    os.mkdir(directory + VOC2007_2012 + JPEGIMAGES)

    fs_2007 = list(map(lambda y: os.path.splitext(y)[0], list(filter(lambda x: x != '.DS_Store', os.listdir(dir_2007 + 'Annotations/')))))
    fs_2012 = list(map(lambda y: os.path.splitext(y)[0], list(filter(lambda x: x != '.DS_Store', os.listdir(dir_2012 + 'Annotations/')))))

    for f_2007 in fs_2007:
        xml = ET.parse(dir_2007 + ANNOTATIONS + f_2007 + '.xml')
        xml.find('filename').text = '2007_' + f_2007 + '.jpg'
        xml.find('folder').text = VOC2007_2012
        xml.write(directory + VOC2007_2012 + ANNOTATIONS + '2007_' + f_2007 + '.xml')
        shutil.copy(dir_2007 + JPEGIMAGES + f_2007 + '.jpg', directory + VOC2007_2012 + JPEGIMAGES + '2007_' + f_2007 + '.jpg')

    for f_2012 in fs_2012:
        xml = ET.parse(dir_2012 + ANNOTATIONS + f_2012 + '.xml')
        xml.find('folder').text = VOC2007_2012
        xml.write(directory + VOC2007_2012 + ANNOTATIONS + f_2012 + '.xml')
        shutil.copy(dir_2012 + JPEGIMAGES + f_2012 + '.jpg', directory + VOC2007_2012 + JPEGIMAGES + f_2012 + '.jpg')

    load_dataset(directory + VOC2007_2012)

join_2007_2012('VOCdevkit/', 'VOCdevkit/VOC2007/', 'VOCdevkit/VOC2012/')
