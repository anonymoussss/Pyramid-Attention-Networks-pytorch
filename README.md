# PAN-pytorch
Well, this repository is forked from [JaveyWang/Pyramid-Attention-Networks-pytorch](https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch) .

based on the original repository code, first I delete the **classifier module code** which the code author said is useless. 

Then, the used dataset is original pascal VOC 2012 which contains 1464 train samples and 1449 val samples, I don't use the 10582-trainaug-images which paper author and code author have used, because in aug website **Semantic Boundaries Dataset and Benchmark**( http://home.bharathh.info/pubs/codes/SBD/download.html ), there is  such a description:

> Please note that the train and val splits included with this dataset are different from the splits in the PASCAL VOC dataset. In particular some "train" images might be part of VOC2012 val.
>
> If you are interested in testing on VOC 2012 val, then use [this train set](http://home.bharathh.info/pubs/codes/SBD/train_noval.txt), which excludes all val images. This was the set used in [our ECCV 2014 paper](http://tinyurl.com/eccv2014-sds). This train set contains 5623 images.

I just want to learn this paper's method.

finally, I try to fix some bugs in original repository.

If you want to use my code, please change the dataset path like :

```
training_data = Voc2012('E:/pascal/VOC2012', 'train',transform=train_transforms)
test_data = Voc2012('E:/pascal/VOC2012', 'val',transform=test_transforms)
```

whichever you are using windows or linux, try 

```
python train.py
```

After my changes applied, run 100 epochs on the **original** VOC 2012 dataset, I get the mIOU 65.13% on val dataset, and the result will  continue to improve obviously if more epochs is applied.





**The following are the original repository readme.md:**

A Pytorch implementation of [Pyramid Attention Networks for Semantic Segmentation](https://arxiv.org/abs/1805.10180) from 2018 paper by Hanchao Li, Pengfei Xiong, Jie An, Lingxue Wang.
![image](https://github.com/JaveyWang/PAN-pytorch/blob/master/results/architecture.png)

# Installation
* Env: Python3.6, [Pytorch1.0-preview](https://pytorch.org/)
* Clone this repository.
* Download the dataset by following the instructions below.

# VOC2012 Dataset
The overall dataset is augmented by Semantic Boundaries Dataset, resulting in training data 10582 and test data 1449. https://www.sun11.me/blog/2018/how-to-use-10582-trainaug-images-on-DeeplabV3-code/

After preparing the data, please change the directory below for training.
```python
training_data = Voc2012('/home/tom/DISK/DISK2/jian/PASCAL/VOC2012', 'train_aug', transform=train_transforms)
test_data = Voc2012('/home/tom/DISK/DISK2/jian/PASCAL/VOC2012', 'val',transform=test_transforms)
```

# Evaluation
![image](https://github.com/JaveyWang/PAN-pytorch/blob/master/results/result.png)

Pixel acc|mIOU
:---------:|:----:
93.19% |78.498%
