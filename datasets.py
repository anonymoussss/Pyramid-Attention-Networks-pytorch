import torch
import torch.utils.data as data
from scipy.misc import imread
from PIL import Image
import numpy as np

class Voc2012(data.Dataset):
    def __init__(self, data_path, trainval="train", transform=None):
        self.data_path = data_path
        self.transform = transform
        self.trainval = trainval

        self.__init_classes()
        #self.names, self.labels = self.__dataset_info()
        self.names = self.__dataset_info()

    def __getitem__(self, index):
        #x = imread(self.data_path + '/JPEGImages/' + self.names[index] + '.jpg', mode='RGB')
        #x = Image.fromarray(x)  # PIL
        x = Image.open(self.data_path + '/JPEGImages/' + self.names[index] + '.jpg')
        x = x.convert('RGB')

        #x_mask = imread(self.data_path + '/SegmentationClass/' + self.names[index] + '.png', mode='L')
        #x_mask = Image.fromarray(x_mask)  # PIL
        x_mask= Image.open(self.data_path + '/SegmentationClass/' + self.names[index] + '.png')

        #y = self.labels[self.names[index]]

        sample = {'image': x, 'label': x_mask}
        if self.transform is not None:
            sample = self.transform(sample)
        x, x_mask = sample['image'], sample['label']
        #y = torch.from_numpy(y)  # 20 classes
        #return x, y, x_mask
        return x, x_mask, self.names[index]

    def __len__(self):
        return len(self.names)

    def __dataset_info(self):
        with open(self.data_path + '/ImageSets/Segmentation/' + self.trainval + '.txt') as f:
            annotations = f.readlines()  # 2913 num pic
        annotations = [n[:-1] for n in annotations]  # delete '\n'
        names = []
        for name in annotations:
            names.append(name)
        #print("the length of ",self.trainval,":",len(names))
        #labels = np.load('/home/liekkas/DISK2/jian/PASCAL/VOC2012/cls_labels.npy')[()]

        #return names, labels
        return names

    def __init_classes(self):
        self.classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))  # assign class_num:0,1,2
        # to each class
        # Change the background label to the last one.
        # self.palette = np.array([20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
