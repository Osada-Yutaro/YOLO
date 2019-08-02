import yolo

data = yolo.pretrain.dataset.Data('../kw_resources/ImageNet/')
size = data.TRAIN_DATA_SIZE
for i in range(size):
    img, _ = data.load_train(i, i + 1)
    print(img.shape)
