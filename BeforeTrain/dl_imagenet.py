import os
import sys
import random as rnd
import tempfile
import requests
import cv2

def download_images(urls_txt, st, ed, imagenet_dir):
    train = open(os.path.join(imagenet_dir, 'train.txt'), 'a')
    validation = open(os.path.join(imagenet_dir, 'validation.txt'), 'a')
    urls = open(urls_txt, 'r', errors='ignore')
    lines = urls.readlines()
    rnd.shuffle(lines)
    count = 0
    length = len(lines)
    for i in range(st, min(length, ed)):
        wnid, url = lines[i].replace('\n', '').split('\t', 1)
        try:
            res = requests.get(url, timeout=3.)
        except:
            continue
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(res.content)
            fp.file.seek(0)
            img = cv2.imread(fp.name)
            if not img is None:
                cv2.imwrite(os.path.join(imagenet_dir, 'Images', str(wnid) + '.png'), img)
                if count%100 == 0:
                    validation.write(str(wnid) + '\n')
                else:
                    train.write(str(wnid) + '\n')
                count += 1
    train.close()
    validation.close()
    urls.close()

if __name__ == '__main__':
    txt = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    imgnet_dir = sys.argv[4]
    download_images(txt, start, end, imgnet_dir)
