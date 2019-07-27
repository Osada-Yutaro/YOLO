import os
import random as rnd
import tempfile
import requests
import cv2

def create_num_url():
    class_num = {}
    with open('ILSVRC2012_classes.txt') as classes:
        lines = classes.readlines()
        for i in range(1000):
            class_num[lines[i].replace('\n', '')] = i

    id_class = {}
    with open('words.txt') as words:
        lines = words.readlines()
        for line in lines:
            wnid, cl = line.replace('\n', '').split('\t')
            id_class[wnid] = cl

    with codecs.open('fall11_urls.txt', 'r', 'utf-8', 'ignore') as urls:
        with open('ILSVRC2012_urls.txt', 'w') as num_urls:
            for line in urls:
                wnid_no, url = line.replace('\n', '').split('\t', 1)
                wnid, _ = wnid_no.split('_')
                cl = id_class[wnid]
                num = class_num.get(cl)
                if not num is None:
                    num_urls.write(str(num) + '\t' + url + '\n')

def download_images():
    train = open('ImageNet/train.txt', 'w')
    validation = open('ImageNet/validation.txt', 'w')
    urls = open('ILSVRC2012_urls.txt', 'r')
    lines = urls.readlines()
    rnd.shuffle(lines)
    count = 0
    for line in lines:
        wnid, url = line.replace('\n', '').split('\t', 1)
        try:
            res = requests.get(url)
        except requests.exceptions.ConnectionError:
            continue
        with tempfile.NamedTemporaryFile(dir='./') as fp:
            fp.write(res.content)
            fp.file.seek(0)
            img = cv2.imread(fp.name)
            if not img is None:
                cv2.imwrite('ImageNet/' + str(wnid) + '_' + str(c) + '.png', img)
                if count%100 == 0:
                    validation.write(str(wnid) + '_' + str(c) + '\n')
                else:
                    train.write(str(wnid) + '_' + str(c) + '\n')
                count += 1
    train.close()
    validation.close()
    urls.close()

if __name__ == '__main__':
    create_num_url()
    download_images()
